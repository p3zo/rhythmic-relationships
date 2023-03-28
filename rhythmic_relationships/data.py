import os

import numpy as np
import pandas as pd
import torch
from rhythmic_relationships import (
    ANNOTATIONS_FILENAME,
    DATASETS_DIR,
    PAIR_LOOKUPS_DIRNAME,
    REPRESENTATIONS_DIRNAME,
)
from rhythmic_relationships.parts import PARTS, get_part_pairs
from rhythmic_relationships.representations import REPRESENTATIONS
from torch.utils.data import Dataset


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
