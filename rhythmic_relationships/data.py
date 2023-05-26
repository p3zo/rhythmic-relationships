import os

import numpy as np
import pandas as pd
import rhythmtoolbox as rtb
import torch
from rhythmic_relationships import (
    ANNOTATIONS_FILENAME,
    DATASETS_DIR,
    PAIR_LOOKUPS_DIRNAME,
    REPRESENTATIONS_DIRNAME,
    REPRESENTATIONS_FILENAME,
)
from rhythmic_relationships.parts import PARTS, get_part_pairs
from rhythmic_relationships.representations import REPRESENTATIONS, DRUM_ROLL_VOICES
from rhythmic_relationships.vocab import get_vocab_encoder_decoder
from torch.utils.data import Dataset
from tqdm import tqdm

PAD_IX = 0


def load_dataset_annotations(dataset_dir):
    """Load the top-level annotations file for a given dataset"""
    df = pd.read_csv(os.path.join(dataset_dir, ANNOTATIONS_FILENAME))
    df.index.name = "roll_id"
    df["filepath"] = df["file_id"].apply(
        lambda x: os.path.join(dataset_dir, REPRESENTATIONS_DIRNAME, f"{x}.npz")
    )
    return df.drop("file_id", axis=1)


def load_repr(segment, repr_ix):
    """Load a representation of a segment"""
    with np.load(segment["filepath"], allow_pickle=True) as npz:
        # TODO: Handle multiple rolls from the same part. For now we just take the first one
        reprs = npz[f"{segment['segment_id']}_{segment['part_id']}"][0]
    return reprs[repr_ix]


def get_seg_fname(filepath, dataset_dir):
    return os.path.splitext(
        filepath.split(os.path.join(dataset_dir, REPRESENTATIONS_DIRNAME))[1]
    )[0].strip("/")


def tokenize_roll(roll, part):
    encode, _ = get_vocab_encoder_decoder(part)

    if part == "Drums":
        if len(roll[0]) != 9:
            raise Exception("Representation must be drum roll for Drums part")
        binarized = (roll > 0).astype(int)
        patterns = ["".join(i.astype(str)) for i in binarized]
        return encode(["start"]) + encode(patterns)

    # Will select the higher note in the case of polyphony
    pitches = roll.argmax(axis=1)

    # Truncate to the standard 88-key piano midi range
    for ix, p in enumerate(pitches):
        if p < 21:
            pitches[ix] = 0
        elif p > 108:
            pitches[ix] = 0

    velocities = roll.max(axis=1)
    velocity_bins = np.array([0.25, 0.5, 0.75, 1])
    binned_velocities = np.digitize(velocities, velocity_bins, right=True)

    pitch_velocities = list(zip(pitches, binned_velocities))

    tokens = []
    for p, v in pitch_velocities:
        if p == 0:
            tokens.append("rest")
            continue
        tokens.append((p, v))

    return encode(["start"]) + encode(tokens)


def get_roll_from_sequence(seq, part):
    """Convert a monophonic sequence of pitches to a piano roll."""
    encode, decode = get_vocab_encoder_decoder(part)

    if seq[0] == encode(["start"])[0]:
        seq = seq[1:]

    roll = np.zeros((len(seq), 128), np.uint8)

    if part == "Drums":
        decoded = decode(seq)
        for tick, token in enumerate(decoded):
            roll[tick, DRUM_ROLL_VOICES] = [int(i) for i in token]
        return roll

    decoded = decode(seq)

    velocity_bins = np.array([0.25, 0.5, 0.75, 1])
    velocity_bin_vals = (velocity_bins * 127).astype(int)

    for tick, token in enumerate(decoded):
        if isinstance(token, str):
            continue

        elif isinstance(token, tuple):
            pitch, velocity_bin = token
            roll[tick, pitch] = velocity_bin_vals[velocity_bin]

        else:
            raise ValueError("Invalid token type")

    return roll


def get_sequences(tokenized, context_len):
    """Partitions a tokenized segment into a recurrent sequence and returns X, Y pairs."""
    X, Y = [], []

    for t in range(len(tokenized)):
        from_ix = 0
        y_from_ix = from_ix
        to_ix = t

        if t == context_len:
            y_from_ix = from_ix + 1
        if t > context_len:
            from_ix = t - context_len
            y_from_ix = from_ix + 1

        context = tokenized[from_ix:to_ix].tolist()
        target = tokenized[y_from_ix : to_ix + 1].tolist()

        if len(context) < context_len:
            c_pad_len = context_len - len(context)
            context = [PAD_IX] * c_pad_len + context

        if len(target) < context_len:
            t_pad_len = context_len - len(target)
            target = [PAD_IX] * t_pad_len + target

        X.append(context)
        Y.append(target)

    return X, Y


def get_pair_sequences(
    p1_tokenized, p2_tokenized, context_len, pad_context=True, pad_target=True
):
    """Partitions a pair of tokenized segments into recurrent sequences and returns X, Y pairs.

    Optionally pad context/target to create pairs that are always the same length.
    """
    X, Y = [], []

    assert isinstance(p1_tokenized, list) and isinstance(p2_tokenized, list)

    for t in range(len(p1_tokenized)):
        from_ix = 0
        to_ix = t + 1

        if t >= context_len:
            from_ix = to_ix - context_len

        context = p1_tokenized[from_ix:to_ix]
        target = p2_tokenized[from_ix:to_ix]

        # Pad context
        if pad_context and len(context) < context_len:
            c_pad_len = context_len - len(context)
            context = [PAD_IX] * c_pad_len + context

        # Pad target
        if pad_target and len(target) < context_len:
            t_pad_len = context_len - len(target)
            target = [PAD_IX] * t_pad_len + target

        X.append(context)
        Y.append(target)

    return X, Y


class PartPairDataset(Dataset):
    """
    Params
        dataset_name, str
            Name of a dataset created by `prepare_dataset.py`.

        part_1, str
            The part to use as the X. See the list of parts in `parts.py`

        part_2, str
            The part to use as the y. See the list of parts in `parts.py`

        repr_1, str
            The representation to use for part 1 segments. See the list of representations in `representations.py`

        repr_2, str
            The representation to use for part 2 segments. See the list of representations in `representations.py`
    """

    def __init__(self, dataset_name, part_1, part_2, repr_1, repr_2):
        if part_1 not in PARTS or part_2 not in PARTS:
            raise ValueError(f"Part names must be one of: {PARTS}")

        if repr_1 not in REPRESENTATIONS or repr_2 not in REPRESENTATIONS:
            raise ValueError(f"Representation names must be one of: {REPRESENTATIONS}")

        self.dataset_dir = os.path.join(DATASETS_DIR, dataset_name)

        # Load the list of available representations
        with open(os.path.join(self.dataset_dir, REPRESENTATIONS_FILENAME), "r") as f:
            self.representations = f.readline().split(",")

        self.repr_1 = repr_1
        self.repr_2 = repr_2
        self.repr_1_ix = self.representations.index(repr_1)
        self.repr_2_ix = self.representations.index(repr_2)

        # Load the part pair metadata
        self.part_1 = part_1
        self.part_2 = part_2

        pair_id = "_".join(get_part_pairs([part_1, part_2])[0])
        pair_lookup_path = os.path.join(
            DATASETS_DIR, dataset_name, PAIR_LOOKUPS_DIRNAME, f"{pair_id}.csv"
        )
        pairs_df = pd.read_csv(pair_lookup_path)

        df = load_dataset_annotations(self.dataset_dir)

        self.p1_pairs = pairs_df.merge(
            df, how="left", left_on=part_1, right_on="roll_id"
        )
        self.p2_pairs = pairs_df.merge(
            df, how="left", left_on=part_2, right_on="roll_id"
        )

        self.loaded_segments = []

    def __len__(self):
        return len(self.p1_pairs)

    def __getitem__(self, idx):
        p1_seg = self.p1_pairs.iloc[idx]
        p2_seg = self.p2_pairs.iloc[idx]

        seg_id = (p1_seg["filepath"], p1_seg["segment_id"])
        self.loaded_segments.append(seg_id)

        p1_seg_repr = load_repr(p1_seg, self.repr_1_ix)
        p2_seg_repr = load_repr(p2_seg, self.repr_2_ix)

        x = torch.from_numpy(p1_seg_repr).to(torch.float32)
        y = torch.from_numpy(p2_seg_repr).to(torch.float32)

        return x, y

    def as_dfs(self, shuffle=False, subset=None):
        """Returns the entire dataset in two dataframes. Useful for analysis.

        :return: Tuple of two dataframes
            pair_df, in which each row is a p1_p2 pair with all descriptors for both parts
            stacked_df: in which the part dfs are vertically stacked
        """

        n_segments = self.__len__()

        p1_pairs = self.p1_pairs.copy()
        p2_pairs = self.p2_pairs.copy()

        if subset:
            p1_pairs = p1_pairs[:subset]
            p2_pairs = p2_pairs[:subset]
            if subset < n_segments:
                n_segments = subset

        print(f"Loading {self.part_1} segments {self.repr_1}")

        segment_ids = []
        filenames = []

        x_reprs = []
        for ix, row in tqdm(p1_pairs.iterrows(), total=n_segments):
            x_reprs.append(load_repr(row, self.repr_1_ix))

            # Only necessary to keep metadata for one part because they are the same
            segment_ids.append(row.segment_id)
            filename = get_seg_fname(row.filepath, self.dataset_dir)
            filenames.append(filename)

        print(f"Loading {self.part_2} segments {self.repr_2}")
        y_reprs = []
        for ix, row in tqdm(p2_pairs.iterrows(), total=n_segments):
            y_reprs.append(load_repr(row, self.repr_2_ix))

        xdf = pd.DataFrame(x_reprs)
        ydf = pd.DataFrame(y_reprs)

        if self.repr_1 == "descriptors":
            xdf.columns = rtb.DESCRIPTOR_NAMES
        if self.repr_2 == "descriptors":
            ydf.columns = rtb.DESCRIPTOR_NAMES

        # Each row is a p1_p2 pair with all descriptors for both parts
        pair_df = xdf.join(ydf, lsuffix=f"_{self.part_1}", rsuffix=f"_{self.part_2}")

        pair_df.insert(0, "segment_id", segment_ids)
        pair_df.insert(0, "filename", filenames)

        xdf["part"] = self.part_1
        ydf["part"] = self.part_2

        xdf["segment_id"] = segment_ids
        ydf["segment_id"] = segment_ids
        xdf["filename"] = filenames
        ydf["filename"] = filenames

        stacked_df = pd.concat([xdf, ydf]).reset_index(drop=True)

        if shuffle:
            return pair_df.sample(frac=1), stacked_df.sample(frac=1)

        return pair_df, stacked_df


class PartDataset(Dataset):
    """
    Params
        dataset_name, str
            Name of a dataset created by `prepare_dataset.py`.

        part, str
            The part to use. See the list of parts in `parts.py`

        representation, str
            The representation to use. See the list of representations in `representations.py`
    """

    def __init__(self, dataset_name, part, representation):
        if part not in PARTS:
            raise ValueError(f"Part must be one of: {PARTS}")

        if representation not in REPRESENTATIONS:
            raise ValueError(f"Representation must be one of: {REPRESENTATIONS}")

        self.part = part

        self.dataset_dir = os.path.join(DATASETS_DIR, dataset_name)

        # Load the list of available representations
        with open(os.path.join(self.dataset_dir, REPRESENTATIONS_FILENAME), "r") as f:
            self.representations = f.readline().split(",")

        self.representation = representation
        self.representation_ix = self.representations.index(representation)

        # Load the segment metadata
        df = load_dataset_annotations(self.dataset_dir)
        self.part_df = df[df.part_id == part]

        self.loaded_segments = []

    def __len__(self):
        return len(self.part_df)

    def __getitem__(self, idx):
        seg = self.part_df.iloc[idx]
        seg_repr = load_repr(seg, self.representation_ix)

        seg_id = (
            get_seg_fname(seg["filepath"], self.dataset_dir) + f'_{seg["segment_id"]}'
        )
        self.loaded_segments.append(seg_id)

        # TODO: remove the astype once the dataset is created w explicit casts
        return torch.from_numpy(seg_repr.astype(np.float32)).to(torch.float32)

    def as_df(self, shuffle=False, subset=None):
        """Returns the entire dataset in a dataframe. Useful for analysis."""
        print(f"Loading {self.part} segment {self.representation}")

        n_segments = self.__len__()

        part_df = self.part_df.copy()

        if subset:
            part_df = part_df[:subset]
            n_segments = subset

        reprs = []
        segment_ids = []
        filenames = []
        for ix, row in tqdm(part_df.iterrows(), total=n_segments):
            reprs.append(load_repr(row, self.representation_ix))
            segment_ids.append(row.segment_id)

            filename = get_seg_fname(row.filepath, self.dataset_dir)
            filenames.append(filename)

        df = pd.DataFrame(reprs)

        if self.representation == "descriptors":
            df.columns = rtb.DESCRIPTOR_NAMES

        df.insert(0, "segment_id", segment_ids)
        df.insert(0, "filename", filenames)

        if shuffle:
            return df.sample(frac=1)

        return df


class PartDatasetSequential(Dataset):
    """
    Loads the same data as a PartDataset, but partitions each segment into a recurrent sequence and returns X, Y pairs.

    Params
        dataset_name, str
            Name of a dataset created by `prepare_dataset.py`.

        part, str
            The part to use. See the list of parts in `parts.py`

        representation, str
            The representation to use. See the list of representations in `representations.py`

        context_len, int
            The length of the context window.

        datasets_dir, str
            The directory where the dataset is stored. Defaults to `DATASETS_DIR`.
    """

    def __init__(
        self, dataset_name, part, representation, context_len, datasets_dir=DATASETS_DIR
    ):
        if part not in PARTS:
            raise ValueError(f"Part must be one of: {PARTS}")

        if representation not in REPRESENTATIONS:
            raise ValueError(f"Representation must be one of: {REPRESENTATIONS}")

        self.part = part

        self.dataset_dir = os.path.join(datasets_dir, dataset_name)

        # Load the list of available representations
        with open(os.path.join(self.dataset_dir, REPRESENTATIONS_FILENAME), "r") as f:
            self.representations = f.readline().split(",")

        self.representation = representation
        self.representation_ix = self.representations.index(representation)

        # Load the segment metadata
        df = load_dataset_annotations(self.dataset_dir)
        self.part_df = df[df.part_id == part]

        self.context_len = context_len

    def __len__(self):
        return len(self.part_df)

    def __getitem__(self, idx):
        seg = self.part_df.iloc[idx]
        seg_repr = load_repr(seg, self.representation_ix)

        tokenized = tokenize_roll(seg_repr, self.part)

        X, Y = get_sequences(tokenized, self.context_len)

        return torch.tensor(X), torch.tensor(Y)


class PartPairDatasetSequential(Dataset):
    """
    Loads the same data as a PartPairDataset, but partitions each segment into a recurrent sequence and returns X, Y pairs.
    """

    def __init__(
        self,
        dataset_name,
        part_1,
        part_2,
        repr_1,
        repr_2,
        context_len,
        datasets_dir=DATASETS_DIR,
    ):
        if part_1 not in PARTS or part_2 not in PARTS:
            raise ValueError(f"Part names must be one of: {PARTS}")

        if repr_1 not in REPRESENTATIONS or repr_2 not in REPRESENTATIONS:
            raise ValueError(f"Representation names must be one of: {REPRESENTATIONS}")

        self.dataset_dir = os.path.join(datasets_dir, dataset_name)

        # Load the list of available representations
        with open(os.path.join(self.dataset_dir, REPRESENTATIONS_FILENAME), "r") as f:
            self.representations = f.readline().split(",")

        self.repr_1 = repr_1
        self.repr_2 = repr_2
        self.repr_1_ix = self.representations.index(repr_1)
        self.repr_2_ix = self.representations.index(repr_2)

        # Load the part pair metadata
        self.part_1 = part_1
        self.part_2 = part_2

        pair_id = "_".join(get_part_pairs([part_1, part_2])[0])
        pair_lookup_path = os.path.join(
            datasets_dir, dataset_name, PAIR_LOOKUPS_DIRNAME, f"{pair_id}.csv"
        )
        pairs_df = pd.read_csv(pair_lookup_path)

        df = load_dataset_annotations(self.dataset_dir)

        self.p1_pairs = pairs_df.merge(
            df, how="left", left_on=part_1, right_on="roll_id"
        )
        self.p2_pairs = pairs_df.merge(
            df, how="left", left_on=part_2, right_on="roll_id"
        )

        self.context_len = context_len

    def __len__(self):
        return len(self.p1_pairs)

    def __getitem__(self, idx):
        p1_seg = self.p1_pairs.iloc[idx]
        p2_seg = self.p2_pairs.iloc[idx]

        p1_seg_repr = load_repr(p1_seg, self.repr_1_ix)
        p2_seg_repr = load_repr(p2_seg, self.repr_2_ix)

        p1_tokenized = tokenize_roll(p1_seg_repr, self.part_1)
        p2_tokenized = tokenize_roll(p2_seg_repr, self.part_2)

        # TODO: rename class from sequential since we're no longer creating sequences
        return torch.LongTensor(p1_tokenized), torch.LongTensor(p2_tokenized)
