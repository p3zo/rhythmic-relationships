import numpy as np
from scipy.spatial import distance_matrix
from rhythmic_relationships.evaluate import compute_oa, compute_kld


def get_flat_nonzero_dissimilarity_matrix(x):
    dists = distance_matrix(x, x, p=2)
    flat = dists.flatten()
    return np.delete(flat, np.arange(0, len(flat), len(x) + 1))


# Descriptors from training dataset
train = np.array(
    [
        [0, 0, 0],
        [0.5, 0, 0],
        [1, 1, 0],
        [0.5, 0.5, 0],
        [1, 1, 1],
    ]
)

# Descriptors from generated dataset
gen = np.array(
    [
        [0, 1, 0],
        [0.5, 1, 0],
        [0, 1, 0],
        [1, 0.5, 0],
        [1, 0.5, 1],
        [1, 0.5, 0],
    ]
)

train_dist = get_flat_nonzero_dissimilarity_matrix(train)

# Stack training and generation
train_gen = np.concatenate((train, gen))
train_gen_dist = get_flat_nonzero_dissimilarity_matrix(train_gen)

# Compute distribution comparison metrics
oa = compute_oa(train_dist, train_gen_dist)
kld = compute_kld(train_dist, train_gen_dist)

print(f"{oa=}, {kld=}")
