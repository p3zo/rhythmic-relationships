import pytest

import torch
from rhythmic_complements.data import PairDataset
from torch.utils.data import DataLoader

DATASET_NAME = "babyslakh_20_1bar_24res"
BATCH_SIZE = 1


def test_dataloaders():
    part_1 = "Drums"
    part_2 = "Guitar"

    with pytest.raises(Exception):
        PairDataset(DATASET_NAME, part_1, part_2, "desc", "hits")

    data = PairDataset(DATASET_NAME, part_1, part_2, "roll", "roll")
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([BATCH_SIZE, 96, 88])
    assert y.size() == torch.Size([BATCH_SIZE, 96, 88])

    data = PairDataset(DATASET_NAME, part_1, part_2, "descriptors", "descriptors")
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([BATCH_SIZE, 18])
    assert y.size() == torch.Size([BATCH_SIZE, 18])

    data = PairDataset(DATASET_NAME, part_1, part_2, "roll", "descriptors")
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([BATCH_SIZE, 96, 88])
    assert y.size() == torch.Size([BATCH_SIZE, 18])

    data = PairDataset(DATASET_NAME, part_1, part_2, "descriptors", "roll")
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([BATCH_SIZE, 18])
    assert y.size() == torch.Size([BATCH_SIZE, 96, 88])

    data = PairDataset(DATASET_NAME, part_1, part_2, "hits", "hits")
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([BATCH_SIZE, 16])
    assert y.size() == torch.Size([BATCH_SIZE, 16])

    data = PairDataset(DATASET_NAME, part_1, part_2, "hits", "roll")
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([BATCH_SIZE, 16])
    assert y.size() == torch.Size([BATCH_SIZE, 96, 88])

    data = PairDataset(DATASET_NAME, part_1, part_2, "descriptors", "hits")
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([BATCH_SIZE, 18])
    assert y.size() == torch.Size([BATCH_SIZE, 16])

    data = PairDataset(DATASET_NAME, part_1, part_2, "pattern", "pattern")
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([BATCH_SIZE, 16])
    assert y.size() == torch.Size([BATCH_SIZE, 16])

    data = PairDataset(DATASET_NAME, part_1, part_2, "roll", "pattern")
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([BATCH_SIZE, 96, 88])
    assert y.size() == torch.Size([BATCH_SIZE, 16])

    data = PairDataset(DATASET_NAME, part_1, part_2, "pattern", "chroma")
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([BATCH_SIZE, 16])
    assert y.size() == torch.Size([BATCH_SIZE, 16, 12])

    data = PairDataset(DATASET_NAME, part_1, part_2, "chroma", "roll")
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([BATCH_SIZE, 16, 12])
    assert y.size() == torch.Size([BATCH_SIZE, 96, 88])
