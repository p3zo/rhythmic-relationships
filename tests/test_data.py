import pytest
import torch
from rhythmic_relationships.data import (
    PartDataset,
    PartPairDataset,
    get_roll_from_sequence,
    tokenize_hits,
    get_sequences,
)
from rhythmic_relationships.vocab import get_vocab_encoder_decoder
from rhythmtoolbox import DESCRIPTOR_NAMES
from torch.utils.data import DataLoader

DATASET_NAME = "babyslakh_20_1bar_4res"
batch_size = 1
PART_1 = "Drums"
PART_2 = "Bass"

N_DESCRIPTORS = len(DESCRIPTOR_NAMES)


def test_get_sequences():
    tokenized = [1, 2, 3, 4]
    ctx, tgt = get_sequences(tokenized, context_len=len(tokenized), pad_ix=0)

    assert ctx == [[0, 0, 0, 0], [1, 0, 0, 0], [1, 2, 0, 0], [1, 2, 3, 0]]
    assert tgt == [[1, 0, 0, 0], [1, 2, 0, 0], [1, 2, 3, 0], [1, 2, 3, 4]]


def get_expected_velocity_from_bin(vel_bin, n_bins):
    return int((vel_bin + 1) * (127 / n_bins))


def test_tokenize_hits():
    # TODO: account for padding token
    n_bins = 4
    hits = [0.0, 0.1, 0.3, 0.6, 1.0]
    assert tokenize_hits(hits, n_bins) == [0, 1, 2, 3, 4]
    hits = [0.0, 0.25, 0.5, 0.75, 1.0]
    assert tokenize_hits(hits, n_bins) == [0, 1, 2, 3, 4]


def test_get_roll_from_sequence():
    tokens = ["start", "rest", (60, 3), (72, 1), "rest"]

    encode, _ = get_vocab_encoder_decoder("Melody")
    sequence = encode(tokens)
    roll = get_roll_from_sequence(sequence, part="Melody")
    assert roll.shape == (4, 128)
    assert roll[0].sum() == 0
    assert roll[3].sum() == 0

    nz10 = roll[1].nonzero()[0]
    assert len(nz10) == 1
    assert nz10[0] == 60
    assert roll[1][nz10[0]] == get_expected_velocity_from_bin(tokens[2][1], n_bins=4)

    nz20 = roll[2].nonzero()[0]
    assert len(nz20) == 1
    assert nz20[0] == 72
    assert roll[2][nz20[0]] == get_expected_velocity_from_bin(tokens[3][1], n_bins=4)

    assert roll[3].sum() == 0

    drum_seq = [1, 2, 11, 101]
    roll = get_roll_from_sequence(drum_seq, part="Drums")
    assert roll.shape == (3, 128)
    assert roll[0].sum() == 0
    assert roll[1].nonzero()[0].tolist() == [47, 51]
    assert roll[2].nonzero()[0].tolist() == [42, 43, 50, 51]


def test_PartDataset():
    with pytest.raises(Exception):
        PartDataset(DATASET_NAME, PART_1, "incorrect-descriptor-name")

    data = PartDataset(DATASET_NAME, PART_1, "descriptors")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x = next(iter(loader))
    assert x.size() == torch.Size([batch_size, N_DESCRIPTORS])

    data = PartDataset(DATASET_NAME, PART_1, "roll")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x = next(iter(loader))
    assert x.size() == torch.Size([batch_size, 16, 128])


def test_PartPairDatasets():
    with pytest.raises(Exception):
        PartPairDataset(
            DATASET_NAME, PART_1, PART_2, "desc", "incorrect-descriptor-name"
        )

    seq_len = 16

    data = PartPairDataset(DATASET_NAME, PART_1, PART_2, "roll", "roll")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, seq_len, 128])
    assert y.size() == torch.Size([batch_size, seq_len, 128])

    data = PartPairDataset(DATASET_NAME, PART_1, PART_2, "descriptors", "descriptors")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, N_DESCRIPTORS])
    assert y.size() == torch.Size([batch_size, N_DESCRIPTORS])

    data = PartPairDataset(DATASET_NAME, PART_1, PART_2, "roll", "descriptors")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, seq_len, 128])
    assert y.size() == torch.Size([batch_size, N_DESCRIPTORS])

    data = PartPairDataset(DATASET_NAME, PART_1, PART_2, "descriptors", "roll")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, N_DESCRIPTORS])
    assert y.size() == torch.Size([batch_size, seq_len, 128])

    data = PartPairDataset(DATASET_NAME, PART_1, PART_2, "hits", "hits")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, seq_len])
    assert y.size() == torch.Size([batch_size, seq_len])

    data = PartPairDataset(DATASET_NAME, PART_1, PART_2, "hits", "roll")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, seq_len])
    assert y.size() == torch.Size([batch_size, seq_len, 128])

    data = PartPairDataset(DATASET_NAME, PART_1, PART_2, "descriptors", "hits")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, N_DESCRIPTORS])
    assert y.size() == torch.Size([batch_size, seq_len])

    data = PartPairDataset(DATASET_NAME, PART_1, PART_2, "pattern", "pattern")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, seq_len])
    assert y.size() == torch.Size([batch_size, seq_len])

    data = PartPairDataset(DATASET_NAME, PART_1, PART_2, "roll", "pattern")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, seq_len, 128])
    assert y.size() == torch.Size([batch_size, seq_len])

    data = PartPairDataset(DATASET_NAME, PART_1, PART_2, "pattern", "chroma")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, seq_len])
    assert y.size() == torch.Size([batch_size, seq_len, 12])

    data = PartPairDataset(DATASET_NAME, PART_1, PART_2, "chroma", "roll")
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    x, y = next(iter(loader))
    assert x.size() == torch.Size([batch_size, seq_len, 12])
    assert y.size() == torch.Size([batch_size, seq_len, 128])
