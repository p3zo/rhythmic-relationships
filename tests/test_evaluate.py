import numpy as np
import pytest

from rhythmic_relationships.evaluate import nucleus, temperatured_softmax


def get_nucleus_pool(probs, p, n_draws=2000):
    """The set of ids `nucleus` can actually return, found by sampling it"""
    return {nucleus(probs.copy(), p) for _ in range(n_draws)}


def test_nucleus_truncates_to_the_smallest_pool_covering_p():
    # Cumulative mass: 0.8, 0.9, 0.95, 0.98, 0.995, 1.0
    probs = np.array([0.80, 0.10, 0.05, 0.03, 0.015, 0.005])

    # p=0.92 is first covered by the top 3, so ids 3, 4 and 5 must be unreachable
    assert get_nucleus_pool(probs, 0.92) == {0, 1, 2}

    # The top token alone covers p=0.5
    assert get_nucleus_pool(probs, 0.5) == {0}


def test_nucleus_pool_grows_with_p():
    rng = np.random.default_rng(13)
    probs = rng.random(354)
    probs /= probs.sum()

    sizes = [len(get_nucleus_pool(probs, p, n_draws=4000)) for p in (0.5, 0.92)]
    assert sizes[0] < sizes[1]

    # A near-uniform distribution over 354 tokens needs most of the mass to reach p=0.92, but
    # never the whole vocabulary -- keeping all but the least likely token is not truncation
    assert sizes[1] < 354


def test_nucleus_never_returns_a_zero_probability_token():
    probs = np.array([0.6, 0.4, 0.0, 0.0])
    assert get_nucleus_pool(probs, 0.92) == {0, 1}


def test_temperatured_softmax_normalises_each_row():
    logits = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    probs = temperatured_softmax(logits, 1.0)
    np.testing.assert_allclose(probs.sum(axis=-1), [1.0, 1.0])

    # A single row of logits still normalises to 1
    np.testing.assert_allclose(temperatured_softmax(logits[0], 1.0).sum(), 1.0)


def test_temperatured_softmax_survives_large_logits():
    """The old implementation overflowed here and fell back to np.float128, which arm64 lacks"""
    logits = np.array([[1000.0, 900.0, 800.0]])
    probs = temperatured_softmax(logits, 0.5)

    assert np.count_nonzero(np.isnan(probs)) == 0
    np.testing.assert_allclose(probs.sum(axis=-1), [1.0])
    assert probs.argmax() == 0


def test_temperatured_softmax_temperature_direction():
    logits = np.array([3.0, 2.0, 1.0])
    cold = temperatured_softmax(logits, 0.5)
    hot = temperatured_softmax(logits, 2.0)

    # Lower temperature concentrates mass on the top logit
    assert cold.max() > hot.max()


def test_temperatured_softmax_matches_a_reference_softmax():
    rng = np.random.default_rng(13)
    logits = rng.normal(size=(4, 32)) * 5
    for temperature in (0.5, 1.0, 1.2):
        expected = np.exp(logits / temperature)
        expected /= expected.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(
            temperatured_softmax(logits, temperature), expected, rtol=1e-10
        )
