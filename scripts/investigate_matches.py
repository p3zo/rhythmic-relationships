"""
Investigations into near matches

Given a bass segment, how many other bass segments are an exact match? for each representation?
- how many other segments are a match regardless of part?
- how many are very similar, like within a small tolerance of each descriptor?
    - Listen to the near matches to see how similar they sound
"""
import collections
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rhythmic_relationships.data import PartPairDataset

dataset_name = "lmdc_3000_2bar_4res"

dataset = PartPairDataset(
    **{
        "dataset_name": dataset_name,
        "part_1": "Bass",
        "part_2": "Melody",
        "repr_1": "hits",
        "repr_2": "hits",
    }
)

# TODO: sqllite db indexed by pattern
# patterns: pattern_id, pattern
# pattern_pairs: pattern_id_1, pattern_id_2

n_segments = len(dataset)
print(f"Loading {n_segments=}...")
loader = DataLoader(dataset, batch_size=n_segments, shuffle=True)
bass_rolls, melody_rolls = next(iter(loader))

products = list(itertools.product(bass_rolls[:1000], bass_rolls[:1000]))

bass_matches = collections.defaultdict(int)
for ix, (r1, r2) in enumerate(tqdm(products)):
    if all(torch.eq(r1 > 1, r2 > 1)):
        bass_matches[ix] += 1

# plt.hist(bass_matches.values())
# plt.savefig("match_hist_bass.png")
# plt.clf()
