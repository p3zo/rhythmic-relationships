import pytest

from rhythmic_relationships.vocab import (
    get_vocab_encoder_decoder,
    get_roll_from_sequence,
)


def test_get_vocab_encoder_decoder():
    itot_expected = [
        ([0, 1, 158, 209, 0], ["rest", "pad", (60, 0), (72, 3), "rest"]),
        (
            [2, 3, 349, 350, 351, 352, 353],
            [(21, 0), (21, 1), (107, 3), (108, 0), (108, 1), (108, 2), (108, 3)],
        ),
    ]
    encoder, decoder = get_vocab_encoder_decoder("Melody")
    for k, v in itot_expected:
        assert decoder(k) == v
        assert encoder(v) == k

    encoder, decoder = get_vocab_encoder_decoder("Bass")
    for k, v in itot_expected:
        assert decoder(k) == v
        assert encoder(v) == k

    # Failure cases
    fail_lists = [[354], [-1]]
    for lst in fail_lists:
        with pytest.raises(Exception):
            decoder(lst)

    # TODO: test Drums as well


def test_get_roll_from_sequence():
    seq = [0, 2, 158, 209, 0]
    roll = get_roll_from_sequence(seq, part="Melody")
    assert roll.shape == (len(seq), 128)
    assert roll[0].sum() == 0
    assert roll[4].sum() == 0

    nz10 = roll[1].nonzero()[0]
    assert len(nz10) == 1
    assert nz10[0] == 21
    assert roll[1][nz10[0]] == 31

    nz20 = roll[2].nonzero()[0]
    assert len(nz20) == 1
    assert nz20[0] == 60
    # Note that the velocity assertions assume 4 velocity bins
    assert roll[2][nz20[0]] == 31

    nz30 = roll[3].nonzero()[0]
    assert len(nz30) == 1
    assert nz30[0] == 72
    assert roll[3][nz30[0]] == 127
