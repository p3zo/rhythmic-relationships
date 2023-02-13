import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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
