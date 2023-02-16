import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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

    def __init__(self, dataset_dir, part_1, part_2, repr_1, repr_2, resolution=24):
        self.dataset_dir = dataset_dir

        self.part_1 = part_1
        self.part_2 = part_2

        self.repr_1 = repr_1
        self.repr_2 = repr_2

        assert repr_1 in ["roll", "pattern", "descriptor"]
        assert repr_2 in ["roll", "pattern", "descriptor"]

        self.df = pd.read_csv(os.path.join(dataset_dir, "rolls.csv"))

        self.pairs_df = pd.read_csv(
            os.path.join(dataset_dir, "pair_lookups", f"{part_1}_{part_2}.csv")
        )

        self.resolution = resolution

    def __len__(self):
        return len(self.pairs_df)

    def __getitem__(self, idx):
        pair_df = pd.DataFrame(self.pairs_df.iloc[idx]).T

        p1_df = pair_df.set_index(self.part_1).join(self.df)
        p2_df = pair_df.set_index(self.part_2).join(self.df)

        p1 = self.load_repr(p1_df, self.repr_1)
        p2 = self.load_repr(p2_df, self.repr_2)

        x = torch.from_numpy(p1).to(torch.float32)
        y = torch.from_numpy(p2).to(torch.float32)

        return x, y

    def load_repr(self, part_pair_df, representation):
        result = []
        for file_id, g in part_pair_df.groupby("file_id"):
            npz_filepath = os.path.join(self.dataset_dir, "rolls", f"{file_id}.npz")
            npz = np.load(npz_filepath)
            seg_part_ids = g["segment_id"].astype(str) + "_" + g["part_id"]
            for seg_part_id in seg_part_ids:
                arr = npz[f"{seg_part_id}_{representation}"]
                # For now take only the first one
                # TODO: handle multiple rolls from the same part in the same segment
                result.append(arr[0])
        return np.array(result)


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
