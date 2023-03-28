import os

import numpy as np
import pandas as pd
import torch
import rhythmtoolbox as rtb
from rhythmic_relationships import (
    ANNOTATIONS_FILENAME,
    DATASETS_DIR,
    PAIR_LOOKUPS_DIRNAME,
    REPRESENTATIONS_DIRNAME,
)
from rhythmic_relationships.parts import PARTS, get_part_pairs
from rhythmic_relationships.representations import REPRESENTATIONS
from torch.utils.data import Dataset
from tqdm import tqdm


def load_dataset_annotations(dataset_name):
    """Load the top-level annotations file for a given dataset"""
    dataset_dir = os.path.join(DATASETS_DIR, dataset_name)
    df = pd.read_csv(os.path.join(dataset_dir, ANNOTATIONS_FILENAME))
    df.index.name = "roll_id"
    df["filepath"] = df["file_id"].apply(
        lambda x: os.path.join(dataset_dir, REPRESENTATIONS_DIRNAME, f"{x}.npz")
    )
    return df.drop("file_id", axis=1)


def load_repr(segment, repr_ix):
    """Load a representation of a segment"""
    npz = np.load(segment["filepath"], allow_pickle=True)
    # TODO: Handle multiple rolls from the same part. For now we just take the first one
    reprs = npz[f"{segment['segment_id']}_{segment['part_id']}"][0]
    return reprs[repr_ix]


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

        self.part_1 = part_1
        self.part_2 = part_2
        self.repr_1 = REPRESENTATIONS.index(repr_1)
        self.repr_2 = REPRESENTATIONS.index(repr_2)

        df = load_dataset_annotations(dataset_name)

        pair_id = "_".join(get_part_pairs([part_1, part_2])[0])
        pair_lookup_path = os.path.join(
            DATASETS_DIR, dataset_name, PAIR_LOOKUPS_DIRNAME, f"{pair_id}.csv"
        )
        pairs_df = pd.read_csv(pair_lookup_path)

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

        p1_seg_repr = load_repr(p1_seg, self.repr_1)
        p2_seg_repr = load_repr(p2_seg, self.repr_2)

        x = torch.from_numpy(p1_seg_repr).to(torch.float32)
        y = torch.from_numpy(p2_seg_repr).to(torch.float32)

        return x, y

    def as_dfs(self):
        """Returns the entire dataset in two dataframes. Useful for analysis.

        :return: Tuple of two dataframes
            pair_df, in which each row is a p1_p2 pair with all descriptors for both parts
            stacked_df: in which the part dfs are vertically stacked
        """
        print(f"Loading {self.part_1} segments")

        x_reprs = []
        x_filenames = []
        for ix, row in tqdm(self.p1_pairs.iterrows(), total=self.__len__()):
            x_reprs.append(load_repr(row, self.repr_1))
            x_filenames.append(os.path.splitext(os.path.basename(row.filepath))[0])

        print(f"Loading {self.part_2} segments")
        y_reprs = []
        y_filenames = []
        for ix, row in tqdm(self.p2_pairs.iterrows(), total=self.__len__()):
            y_reprs.append(load_repr(row, self.repr_2))
            y_filenames.append(os.path.splitext(os.path.basename(row.filepath))[0])

        xdf = pd.DataFrame(x_reprs)
        ydf = pd.DataFrame(y_reprs)

        if self.repr_1 == REPRESENTATIONS.index("descriptors"):
            xdf.columns = rtb.DESCRIPTOR_NAMES
            xdf["noi"] = xdf["noi"].fillna(0).astype(int)
            xdf["polysync"] = xdf["polysync"].fillna(0).astype(int)

        if self.repr_2 == REPRESENTATIONS.index("descriptors"):
            ydf.columns = rtb.DESCRIPTOR_NAMES
            ydf["noi"] = ydf["noi"].fillna(0).astype(int)
            ydf["polysync"] = ydf["polysync"].fillna(0).astype(int)

        xdf["filename"] = x_filenames
        ydf["filename"] = y_filenames

        # Each row is a p1_p2 pair with all descriptors for both parts
        pair_df = xdf.join(ydf, lsuffix=self.part_1, rsuffix=self.part_2)

        xdf["part"] = self.part_1
        ydf["part"] = self.part_2
        stacked_df = pd.concat([xdf, ydf]).reset_index(drop=True)

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
        self.representation = REPRESENTATIONS.index(representation)

        df = load_dataset_annotations(dataset_name)
        self.part_df = df[df.part_id == part]

    def __len__(self):
        return len(self.part_df)

    def __getitem__(self, idx):
        seg = self.part_df.iloc[idx]
        seg_repr = load_repr(seg, self.representation)
        return torch.from_numpy(seg_repr).to(torch.float32)

    def as_df(self):
        """Returns the entire dataset in a dataframe. Useful for analysis."""
        print(f"Loading {self.part} segment {REPRESENTATIONS[self.representation]}")
        reprs = []
        filenames = []
        for ix, row in tqdm(self.part_df.iterrows(), total=self.__len__()):
            reprs.append(load_repr(row, self.representation))
            filenames.append(os.path.splitext(os.path.basename(row.filepath))[0])
        df = pd.DataFrame(reprs)
        if self.representation == "descriptors":
            df.columns = rtb.DESCRIPTOR_NAMES
        df["filename"] = filenames
        return df
