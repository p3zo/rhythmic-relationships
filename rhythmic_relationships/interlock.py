"""How two parts sit against each other, and a score for whether a pairing looks real.

The mashup ablation found that a generated rhythm adds nothing to the search for a partner
segment: conditioning it on the input melody or on someone else's produced relationships equally
close to real ones, and a segment drawn at random did as well. Retrieval on the relationship
itself does much better, but only on the right features. Shown a melody, its real partner and an
imposter drawn from another melody's real partners, the two paired descriptors from the thesis
prefer the real one 0.574 of the time, while the features here prefer it 0.683.

What separates them is that these read where the two parts' onsets fall relative to each other,
step by step, rather than summarising each part and comparing the summaries. Of a real bass's
onsets, 57.6% land on a melody onset, against 46.3% once partners are reassigned at random.

Both figures are sensitive to how the pairs were sampled, which is why load_co_occurring_hits
takes a per-file cap. Measured on 3,000 pairs from 3,000 songs these read 0.574 and 0.683; on 500
pairs taken in order, which is ten songs, the same code reads 0.392 and 0.737.

See scripts/relationship_retrieval.py, which measures all of this, and docs/index.html, which
ports the two functions here into the browser.
"""

import numpy as np


def interlock_features(a, b):
    """Where the two parts' onsets fall relative to each other, as rates rather than counts.

    Counts would be dominated by how busy the two parts happen to be; these are shares, so two
    parts that interlock the same way score the same however dense they are.

    `a` is the part being played against and `b` the candidate partner; the two are not
    interchangeable, since two of the three features are shares of one part or the other.

    How much of each part lands on the beat was here too, and is not, because it is a property of
    one part on its own: reassigning partners at random leaves both means unmoved, and dropping
    the pair of them costs 0.001 of lineup accuracy. "The share filling A's gaps" is also absent,
    being one minus the first of these; including it made the feature covariance singular and the
    scores that read it meaningless.
    """
    on_a, on_b = a > 0, b > 0
    steps = a.shape[1]
    n_a = on_a.sum(axis=1).clip(1)
    n_b = on_b.sum(axis=1).clip(1)
    return np.stack([
        (on_a & on_b).sum(axis=1) / n_b,          # of B's onsets, the share landing with A
        (on_a & ~on_b).sum(axis=1) / n_a,         # of A's onsets, the share B leaves alone
        (~on_a & ~on_b).sum(axis=1) / steps,      # how much of the bar neither part touches
    ], axis=1)


def usable(a, b):
    """Rows where both parts sound. No feature here means anything against silence."""
    return ((a > 0).sum(axis=1) > 0) & ((b > 0).sum(axis=1) > 0)


class PairScore:
    """How much a pairing looks like one that was actually recorded together.

    Learned by contrast, not by typicality. Scoring closeness to the average real relationship
    turns out to prefer *shuffled* pairs, because reassignment averages out the idiosyncrasies a
    real pairing has; the question is not "is this a typical relationship" but "is this pairing
    real or arbitrary". So the score is a linear discriminant fitted on real pairs against
    shuffled ones, which is the same closed form either way but pointed at the right contrast.

    With one pooled covariance and the two classes equally likely, the score is the log odds that
    the pairing is real rather than arbitrary, so a logistic of it reads as a probability.
    """

    def __init__(self, features_real, features_shuffled):
        mu1, mu0 = features_real.mean(axis=0), features_shuffled.mean(axis=0)
        pooled = np.cov(np.concatenate([features_real - mu1, features_shuffled - mu0]),
                        rowvar=False)
        if pooled.ndim == 0:
            pooled = pooled.reshape(1, 1)
        self.w = np.linalg.solve(pooled + 1e-9 * np.eye(len(pooled)), mu1 - mu0)
        self.offset = float(self.w @ (mu1 + mu0) / 2)

    def score(self, features):
        return features @ self.w - self.offset


def lineup_accuracy(model, a, b, rng, n_imposters=40, features=interlock_features):
    """How often the score prefers a real partner to an imposter, over `n_imposters` rounds.

    Imposters are other melodies' real partners, not segments from the general pool. Drawing them
    from the pool would let the score win on population - segments that co-occur with a melody at
    all are not a random sample of the part - and that has nothing to do with who plays with whom.
    Shuffling real partners leaves exactly one difference between the two candidates: which
    melody this one was actually recorded with.
    """
    keep_real = usable(a, b)
    real_score = np.full(len(a), np.nan)
    real_score[keep_real] = model.score(features(a[keep_real], b[keep_real]))

    wins, total = 0.0, 0
    for _ in range(n_imposters):
        # A derangement is not required: a melody drawing its own partner is a genuine tie and
        # the 0.5 credit below scores it as one
        picks = rng.permutation(len(b))
        cand = b[picks]
        keep = usable(a, cand) & keep_real
        if not keep.any():
            continue
        got = model.score(features(a[keep], cand[keep]))
        diff = real_score[keep] - got
        wins += float((diff > 0).sum() + 0.5 * (diff == 0).sum())
        total += int(keep.sum())
    return wins / max(total, 1), total


def fit_pair_score(a, b, seed, features=interlock_features):
    """A scorer for one directed part pair, from segments recorded together.

    The negative class is the same partners reassigned to other inputs, so the only difference
    between the two classes is who was playing with whom.
    """
    keep = usable(a, b)
    a, b = a[keep], b[keep]
    shuffled = b[np.random.default_rng(seed).permutation(len(b))]
    return PairScore(features(a, b), features(a, shuffled))
