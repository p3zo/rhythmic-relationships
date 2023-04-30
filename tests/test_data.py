import pytest
import torch
from rhythmtoolbox import DESCRIPTOR_NAMES
from rhythmic_relationships.data import PartDataset, PartPairDataset
from torch.utils.data import DataLoader

dataset_name = "babyslakh_20_1bar_4res"
batch_size = 1
part_1 = "Drums"
part_2 = "Bass"

N_DESCRIPTORS = len(DESCRIPTOR_NAMES)


def test_PartDataset():
    with pytest.raises(Exception):
        PartDataset(dataset_name, part_1, "incorrect-descriptor-name")

    data = PartDataset(dataset_name, part_1, "descriptors")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x = next(iter(loader))
    assert x.size() == torch.Size([batch_size, N_DESCRIPTORS])

    data = PartDataset(dataset_name, part_1, "roll")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x = next(iter(loader))
    assert x.size() == torch.Size([batch_size, 16, 128])


def test_PartPairDatasets():
    with pytest.raises(Exception):
        PartPairDataset(
            dataset_name, part_1, part_2, "desc", "incorrect-descriptor-name"
        )

    data = PartPairDataset(dataset_name, part_1, part_2, "roll", "roll")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, 16, 128])
    assert y.size() == torch.Size([batch_size, 16, 128])

    data = PartPairDataset(dataset_name, part_1, part_2, "descriptors", "descriptors")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, N_DESCRIPTORS])
    assert y.size() == torch.Size([batch_size, N_DESCRIPTORS])

    data = PartPairDataset(dataset_name, part_1, part_2, "roll", "descriptors")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, 16, 128])
    assert y.size() == torch.Size([batch_size, N_DESCRIPTORS])

    data = PartPairDataset(dataset_name, part_1, part_2, "descriptors", "roll")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, N_DESCRIPTORS])
    assert y.size() == torch.Size([batch_size, 16, 128])

    data = PartPairDataset(dataset_name, part_1, part_2, "hits", "hits")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, 16])
    assert y.size() == torch.Size([batch_size, 16])

    data = PartPairDataset(dataset_name, part_1, part_2, "hits", "roll")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, 16])
    assert y.size() == torch.Size([batch_size, 16, 128])

    data = PartPairDataset(dataset_name, part_1, part_2, "descriptors", "hits")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, N_DESCRIPTORS])
    assert y.size() == torch.Size([batch_size, 16])

    data = PartPairDataset(dataset_name, part_1, part_2, "pattern", "pattern")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, 16])
    assert y.size() == torch.Size([batch_size, 16])

    data = PartPairDataset(dataset_name, part_1, part_2, "roll", "pattern")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, 16, 128])
    assert y.size() == torch.Size([batch_size, 16])

    data = PartPairDataset(dataset_name, part_1, part_2, "pattern", "chroma")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, 16])
    assert y.size() == torch.Size([batch_size, 16, 12])

    data = PartPairDataset(dataset_name, part_1, part_2, "chroma", "roll")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, 16, 12])
    assert y.size() == torch.Size([batch_size, 16, 128])
