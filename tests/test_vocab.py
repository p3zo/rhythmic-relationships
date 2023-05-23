import pytest

from rhythmic_relationships.vocab import get_vocab_encoder_decoder


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
