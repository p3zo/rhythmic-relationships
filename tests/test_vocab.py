import pytest
from rhythmic_relationships.vocab import get_vocab_encoder_decoder


def test_get_vocab_encoder_decoder():
    encode, decode = get_vocab_encoder_decoder("Melody")

    tokens = ["start", "rest", (60, 3), (72, 1), "rest"]

    assert encode(tokens) == [0, 1, 161, 207, 1]

    itot_expected = [
        ([0, 1, 158, 209, 1], ["start", "rest", (60, 0), (72, 3), "rest"]),
        (
            [1, 2, 3, 349, 350, 351, 352, 353],
            [
                "rest",
                (21, 0),
                (21, 1),
                (107, 3),
                (108, 0),
                (108, 1),
                (108, 2),
                (108, 3),
            ],
        ),
    ]

    for k, v in itot_expected:
        assert decode(k) == v
        assert encode(v) == k

    encode, decode = get_vocab_encoder_decoder("Bass")
    for k, v in itot_expected:
        assert decode(k) == v
        assert encode(v) == k

    # Failure cases
    fail_lists = [[354], [-1]]
    for lst in fail_lists:
        with pytest.raises(Exception):
            decode(lst)

    encode, decode = get_vocab_encoder_decoder("Drums")

    tokens = ["start", "111101000", "111111111"]
    assert encode(tokens) == [0, 489, 512]

    # Failure cases
    fail_lists = [[514], [-1]]
    for lst in fail_lists:
        with pytest.raises(Exception):
            decode(lst)
