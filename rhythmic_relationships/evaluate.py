import numpy as np
from scipy import stats, integrate
from scipy.spatial import distance_matrix


def get_flat_nonzero_dissimilarity_matrix(x):
    dists = distance_matrix(x, x, p=2)
    flat = dists.flatten()
    return np.delete(flat, np.arange(0, len(flat), len(x) + 1))


def compute_oa(A, B):
    print("Computing OA")

    # obtain density functions from data series
    pdf_A = stats.gaussian_kde(A)
    pdf_B = stats.gaussian_kde(B)

    lower_limit = np.min((np.min(A), np.min(B)))  # min of both
    upper_limit = np.max((np.max(A), np.max(B)))  # max of both

    # TODO: subsample pdfs to speed up integration
    area = integrate.quad(lambda x: min(pdf_A(x), pdf_B(x)), lower_limit, upper_limit)
    return area[0]


def compute_kld(A, B, num_sample=1000):
    print("Computing KLD")

    pdf_A = stats.gaussian_kde(A)
    pdf_B = stats.gaussian_kde(B)

    sample_A = np.linspace(np.min(A), np.max(A), num_sample)
    sample_B = np.linspace(np.min(B), np.max(B), num_sample)

    return stats.entropy(pdf_A(sample_A), pdf_B(sample_B))


def compute_oa_and_kld(train_dist, train_gen_dist):
    oa = compute_oa(train_dist, train_gen_dist)
    kld = compute_kld(train_dist, train_gen_dist)
    return oa, kld
