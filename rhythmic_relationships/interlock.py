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

Being able to tell a real pairing from an arbitrary one is not the same as being able to choose
one, and the difference decides what retrieval should optimise. "Which candidate is most likely a
real partner rather than an arbitrary segment" is answered by whichever candidate doubles the
input exactly: real partners share onsets more often than arbitrary ones, so more sharing is
always stronger evidence, and among 30,000 candidates something always sits at the extreme. That
is a property of the objective and not of the model - a linear discriminant, a quadratic one and a
nearest-neighbour estimate of the same ratio all put around 40% unisons in their top ten. What a
mashup wants instead is a pairing drawn from the population of real ones, which is a question about
matching a distribution rather than maximising a ratio: see fit_relationship_targets.

See scripts/relationship_retrieval.py, which measures all of this, and docs/index.html, which
ports these functions into the browser.
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


# A partner every one of whose onsets lands on one of the input's adds nothing rhythmically. It is
# not rare - between 18% and 39% of real pairs are exactly this, depending on the parts - but it is
# the one thing a mashup must not be, so these are the relationships retrieval does not aim at.
# Nothing sits between 0.95 and 1.0 in the data, so this excludes exact doubles and nothing else.
MAX_TOGETHER = 0.95


def fit_relationship_targets(a, b, n_targets, seed, features=interlock_features):
    """Relationships real pairs of these parts had, and the metric for comparing against them.

    Retrieval aims at one of these rather than at the maximum of a score. Every target is a
    relationship that some real pair actually stood in, so a candidate close to one is
    complementary in a way real music is: measured over 50 inputs, the segments retrieved this way
    cover 53% of the input's onsets and leave 42% of them alone, against 58% and 40% for real
    pairs, where maximising the discriminator covers 85% and leaves 19%.

    Sampling a target rather than aiming at the average also keeps the results varied. The average
    relationship is one point, so every input walks toward the same few segments - 18 distinct
    rhythms across 50 inputs, against 369 when the target is drawn.

    Distances are Mahalanobis under the real pairs' own covariance, so "close" means close relative
    to how much real relationships vary in that direction.

    The covariance and the reported means describe every real pair, doubles included, since they
    are what the page compares a result against; only the targets are drawn from the rest.
    """
    keep = usable(a, b)
    f = features(a[keep], b[keep])
    precision = np.linalg.inv(np.cov(f, rowvar=False) + 1e-9 * np.eye(f.shape[1]))

    worth_aiming_at = f[f[:, 0] < MAX_TOGETHER]
    rng = np.random.default_rng(seed)
    targets = worth_aiming_at[
        rng.choice(len(worth_aiming_at), size=min(n_targets, len(worth_aiming_at)), replace=False)
    ]
    return targets, precision, f.mean(axis=0)


def relationship_distance(features, target, precision):
    """How far a pairing sits from a target relationship, in units of how much real pairs vary."""
    d = features - target
    return np.einsum("ij,jk,ik->i", d, precision, d)


def fit_pair_score(a, b, seed, features=interlock_features):
    """A scorer for one directed part pair, from segments recorded together.

    The negative class is the same partners reassigned to other inputs, so the only difference
    between the two classes is who was playing with whom.
    """
    keep = usable(a, b)
    a, b = a[keep], b[keep]
    shuffled = b[np.random.default_rng(seed).permutation(len(b))]
    return PairScore(features(a, b), features(a, shuffled))
