"""
Investigations into near matches

Given a bass segment, how many other bass segments are an exact match? for each representation?
- how many other segments are a match regardless of part?
- how many are very similar, like within a small tolerance of each descriptor?
    - Listen to the near matches to see how similar they sound

For matches, what is the distribution of descriptors in paired drum segments?
- Sample from that distribution
- Select the drum segment with the closest euclidean distance

What other models are appropriate for selecting existing segments rather than generate new ones?
"""
import collections

import matplotlib.pyplot as plt
import numpy as np
from rhythmic_relationships.data import PartPairDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

dataset_name = "lmdc_1000_1bar_4res"

bass_piano_dataset = PartPairDataset(
    **{
        "dataset_name": dataset_name,
        "part_1": "Bass",
        "part_2": "Piano",
        "repr_1": "roll",
        "repr_2": "roll",
    }
)

n_segments = len(bass_piano_dataset)
print(f"Loading {n_segments=}...")
bass_piano_loader = DataLoader(bass_piano_dataset, batch_size=n_segments, shuffle=True)
bass_rolls, piano_rolls = next(iter(bass_piano_loader))

exact_matches_per_roll = collections.defaultdict(int)
for ix, b_roll in enumerate(tqdm(bass_rolls[:100])):
    for r in bass_rolls:
        if np.array_equal(b_roll, r):
            exact_matches_per_roll[ix] += 1

plt.hist(exact_matches_per_roll.values())
plt.savefig("match_hist.png")
plt.clf()
