"""
Question: What % of segments are at 4 ticks per beat?

To answer that, this script loads the same segment at 2 resolutions (4 and 24), scales the 24 down to 4, and checks
if the two are equal.

Results:
- In babyslakh_1bar_24res_20 compared to 4res: 2481 / 13509 (18.4%) segments have no difference
- In lmb_clean_1bar_24res_593 compared to 4res: 67357 / 393710 (17.1 %) segments have no difference
"""

import glob
import os

from prepare_data import slice_midi_file
from scipy import ndimage
from tqdm import tqdm

dataset_dir = "../input/lmd_clean"
filepaths = glob.glob(os.path.join(dataset_dir, "**/*.mid"), recursive=True)
from_res = 24
to_res = 4
factor = to_res / from_res

n_equals = 0
n_segments = 0
subset = 1000
for filepath in tqdm(filepaths[:subset]):
    slices_from, _ = slice_midi_file(filepath, resolution=from_res)
    slices_to, _ = slice_midi_file(filepath, resolution=to_res)

    if slices_from is None:
        continue

    for seg_part_key in slices_from:
        for spix, sp_roll in enumerate(slices_from[seg_part_key]):
            n_segments += 1
            roll_from = sp_roll[0]
            try:
                roll_to = slices_to[seg_part_key][spix][0]
            except IndexError:
                continue

            downsampled = ndimage.zoom(roll_from, (factor, 1), order=0)
            assert downsampled.shape == roll_to.shape

            n_diffs = len(roll_to[roll_to != downsampled])
            print(filepath, spix, n_diffs)

            if n_diffs == 0:
                n_equals += 1

print(f"{n_equals}=")
print(f"{n_segments}=")
print(f"{n_equals / n_segments}=")
