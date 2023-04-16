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
from rhythmic_relationships.representations import REPRESENTATIONS
from torch.utils.data import Dataset
from tqdm import tqdm


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


def split_filepath_on_dirname(filepath, dirname):
    return os.path.splitext(filepath.split(dirname)[1])[0].strip("/")


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

    def __len__(self):
        return len(self.p1_pairs)

    def __getitem__(self, idx):
        p1_seg = self.p1_pairs.iloc[idx]
        p2_seg = self.p2_pairs.iloc[idx]

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
            filename = split_filepath_on_dirname(
                row.filepath, os.path.join(self.dataset_dir, REPRESENTATIONS_DIRNAME)
            )
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

    def __len__(self):
        return len(self.part_df)

    def __getitem__(self, idx):
        seg = self.part_df.iloc[idx]
        seg_repr = load_repr(seg, self.representation_ix)
        return torch.from_numpy(seg_repr).to(torch.float32)

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

            filename = split_filepath_on_dirname(
                row.filepath, os.path.join(self.dataset_dir, REPRESENTATIONS_DIRNAME)
            )
            filenames.append(filename)

        df = pd.DataFrame(reprs)

        if self.representation == "descriptors":
            df.columns = rtb.DESCRIPTOR_NAMES

        df.insert(0, "segment_id", segment_ids)
        df.insert(0, "filename", filenames)

        if shuffle:
            return df.sample(frac=1)

        return df
