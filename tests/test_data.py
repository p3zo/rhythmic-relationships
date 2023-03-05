import pytest
import torch
from torch.utils.data import DataLoader

from rhythmic_complements.data import PairDataset

dataset_name = "babyslakh_20_1bar_4res"
batch_size = 1
part_1 = "Drums"
part_2 = "Guitar"


def test_dataloaders():
    with pytest.raises(Exception):
        PairDataset(dataset_name, part_1, part_2, "desc", "incorrect-descriptor-name")

    data = PairDataset(dataset_name, part_1, part_2, "roll", "roll")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, 16, 128])
    assert y.size() == torch.Size([batch_size, 16, 128])

    data = PairDataset(dataset_name, part_1, part_2, "descriptors", "descriptors")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, 18])
    assert y.size() == torch.Size([batch_size, 18])

    data = PairDataset(dataset_name, part_1, part_2, "roll", "descriptors")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, 16, 128])
    assert y.size() == torch.Size([batch_size, 18])

    data = PairDataset(dataset_name, part_1, part_2, "descriptors", "roll")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, 18])
    assert y.size() == torch.Size([batch_size, 16, 128])

    data = PairDataset(dataset_name, part_1, part_2, "hits", "hits")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, 16])
    assert y.size() == torch.Size([batch_size, 16])

    data = PairDataset(dataset_name, part_1, part_2, "hits", "roll")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, 16])
    assert y.size() == torch.Size([batch_size, 16, 128])

    data = PairDataset(dataset_name, part_1, part_2, "descriptors", "hits")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, 18])
    assert y.size() == torch.Size([batch_size, 16])

    data = PairDataset(dataset_name, part_1, part_2, "pattern", "pattern")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, 16])
    assert y.size() == torch.Size([batch_size, 16])

    data = PairDataset(dataset_name, part_1, part_2, "roll", "pattern")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, 16, 128])
    assert y.size() == torch.Size([batch_size, 16])

    data = PairDataset(dataset_name, part_1, part_2, "pattern", "chroma")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, 16])
    assert y.size() == torch.Size([batch_size, 16, 12])

    data = PairDataset(dataset_name, part_1, part_2, "chroma", "roll")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, 16, 12])
    assert y.size() == torch.Size([batch_size, 16, 128])
