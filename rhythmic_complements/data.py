import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

REPRESENTATIONS = ["roll", "pattern", "descriptor"]


class PairDataset(Dataset):
    """
    Params
        annotations_filepath, str
            Path to an output directory created by `prepare_data.py`.

        part_1, str
            The part to use as the X. See the list of parts in `prepare_data.py`

        part_2, str
            The part to use as the y. See the list of parts in `prepare_data.py`

        repr_1, str
            The representation to use for part 1 segments.

        repr_1, str
            The representation of part 1.
    """

    def __init__(self, dataset_dir, part_1, part_2, repr_1, repr_2):
        self.dataset_dir = dataset_dir

        self.part_1 = part_1
        self.part_2 = part_2

        assert repr_1 in REPRESENTATIONS
        assert repr_2 in REPRESENTATIONS

        self.repr_1 = REPRESENTATIONS.index(repr_1)
        self.repr_2 = REPRESENTATIONS.index(repr_2)

        df = pd.read_csv(os.path.join(dataset_dir, "rolls.csv"))
        df.index.name = "roll_id"
        df["filepath"] = df["file_id"].apply(
            lambda x: os.path.join(dataset_dir, "rolls", f"{x}.npz")
        )
        df = df.drop("file_id", axis=1)

        pairs_df = pd.read_csv(
            os.path.join(dataset_dir, "pair_lookups", f"{part_1}_{part_2}.csv")
        )

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


class SegrollDataset(Dataset):
    def __init__(self, filepath):
        npz = np.load(filepath)
        print(f"Loading {filepath}...")
        self.segments = npz["segrolls"]

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return torch.from_numpy(self.segments[idx]).to(torch.float32)


class DescriptorDataset(Dataset):
    def __init__(self, filepath):
        df = pd.read_csv(filepath)
        df = df.drop(["segment_id", "filepath"], axis=1)
        df = df.dropna()

        # Rescale to [0, 1]
        rescaled = (df - df.min()) / (df.max() - df.min())

        self.x = torch.from_numpy(rescaled.values).to(torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


class DescriptorPairsDataset(Dataset):
    def __init__(self, filepath, part_1, part_2):
        df = pd.read_csv(filepath)
        df = df.dropna()

        # Rescale to [0, 1]
        rescaled = (df - df.min()) / (df.max() - df.min())

        p1 = rescaled[[c for c in rescaled.columns if part_1 in c]].values
        p2 = rescaled[[c for c in rescaled.columns if part_2 in c]].values

        self.x = torch.from_numpy(p1).to(torch.float32)
        self.y = torch.from_numpy(p2).to(torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
