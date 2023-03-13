import os

import numpy as np
import pandas as pd
import torch
from rhythmic_complements import (
    ANNOTATIONS_FILENAME,
    DATASETS_DIR,
    PAIR_LOOKUPS_DIRNAME,
    REPRESENTATIONS_DIRNAME,
)
from rhythmic_complements.parts import get_part_pairs
from rhythmic_complements.representations import REPRESENTATIONS
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


class PairDataset(Dataset):
    """
    Params
        annotations_filepath, str
            Path to an output directory created by `prepare_dataset.py`.

        part_1, str
            The part to use as the X. See the list of parts in `prepare_dataset.py`

        part_2, str
            The part to use as the y. See the list of parts in `prepare_dataset.py`

        repr_1, str
            The representation to use for part 1 segments.

        repr_1, str
            The representation of part 1.
    """

    def __init__(self, dataset_name, part_1, part_2, repr_1, repr_2):
        self.part_1 = part_1
        self.part_2 = part_2

        if repr_1 not in REPRESENTATIONS or repr_2 not in REPRESENTATIONS:
            raise ValueError(f"Representation names must be one of: {REPRESENTATIONS}")

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
        p1 = self.p1_pairs.iloc[idx]
        p2 = self.p2_pairs.iloc[idx]

        p1_repr = self.load_repr(p1, self.repr_1)
        p2_repr = self.load_repr(p2, self.repr_2)

        x = torch.from_numpy(p1_repr).to(torch.float32)
        y = torch.from_numpy(p2_repr).to(torch.float32)

        return x, y

    def load_repr(self, pair, repr_ix):
        npz = np.load(pair["filepath"], allow_pickle=True)
        # TODO: Handle multiple rolls from the same part. For now we just take the first one
        reprs = npz[f"{pair['segment_id']}_{pair['part_id']}"][0]
        return reprs[repr_ix]
