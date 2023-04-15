"""Select segments with at least n parts from a dataset at random."""

import os

import numpy as np
from rhythmic_relationships import DATASETS_DIR
from rhythmic_relationships.data import load_dataset_annotations
from rhythmic_relationships.io import get_pmid_segment, load_midi_file

INPUT_DATA_DIR = "../input/lmd_clean"

dataset_name = "lmdc_250_1bar_4res"
representation = "roll"
n_parts_min = 2

# Load dataset annotations
df = load_dataset_annotations(os.path.join(DATASETS_DIR, dataset_name))
df = df.sample(frac=1)  # shuffle

# Choose a random segment with more than `n_parts_min` parts
n_parts_by_seg = df.groupby(["segment_id", "filepath"]).apply(len)

output_dir = os.path.join("output", "midi")
for n in range(5):
    seg_id, fp = np.random.choice(n_parts_by_seg[n_parts_by_seg >= n_parts_min].index)
    seg_df = df.loc[(df.filepath == fp) & (df.segment_id == seg_id)]

    track_name = os.path.splitext("/".join(fp.split("/")[-2:]))[0]
    pmid = load_midi_file(os.path.join(INPUT_DATA_DIR, f"{track_name}.mid"))

    parts = seg_df.part_id.values.tolist()
    pmid_slice = get_pmid_segment(pmid, segment_num=seg_id, seg_size=2, parts=parts)

    file_seg_id = f"{os.path.splitext(os.path.basename(fp))[0]}_{seg_id}"
    print(f"Selected segment {file_seg_id}\n\tParts: {parts}\n")

    pmid_slice.write(os.path.join(output_dir, f"{file_seg_id}.mid"))
