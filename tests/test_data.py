import pytest

import torch
from rhythmic_complements.data import PairDataset
from torch.utils.data import DataLoader

DATASET_DIR = "../output/lmd_clean_1bar_24res_500"
BATCH_SIZE = 1


def test_dataloaders():
    part_1 = "Drums"
    part_2 = "Guitar"

    with pytest.raises(Exception):
        PairDataset(DATASET_DIR, part_1, part_2, "desc", "pattern")

    data = PairDataset(DATASET_DIR, part_1, part_2, "roll", "roll")
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([BATCH_SIZE, 1, 96, 88])
    assert y.size() == torch.Size([BATCH_SIZE, 1, 96, 88])

    data = PairDataset(DATASET_DIR, part_1, part_2, "descriptor", "descriptor")
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([BATCH_SIZE, 1, 18])
    assert y.size() == torch.Size([BATCH_SIZE, 1, 18])

    data = PairDataset(DATASET_DIR, part_1, part_2, "roll", "descriptor")
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([BATCH_SIZE, 1, 96, 88])
    assert y.size() == torch.Size([BATCH_SIZE, 1, 18])

    data = PairDataset(DATASET_DIR, part_1, part_2, "descriptor", "roll")
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([BATCH_SIZE, 1, 18])
    assert y.size() == torch.Size([BATCH_SIZE, 1, 96, 88])

    data = PairDataset(DATASET_DIR, part_1, part_2, "pattern", "pattern")
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([BATCH_SIZE, 1, 16])
    assert y.size() == torch.Size([BATCH_SIZE, 1, 16])

    data = PairDataset(DATASET_DIR, part_1, part_2, "pattern", "roll")
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([BATCH_SIZE, 1, 16])
    assert y.size() == torch.Size([BATCH_SIZE, 1, 96, 88])

    data = PairDataset(DATASET_DIR, part_1, part_2, "descriptor", "pattern")
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([BATCH_SIZE, 1, 18])
    assert y.size() == torch.Size([BATCH_SIZE, 1, 16])
