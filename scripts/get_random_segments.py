"""Select segments with at least n parts from a dataset at random."""
import os
import numpy as np
from rhythmic_relationships.data import load_dataset_annotations
from rhythmic_relationships.io import write_midi_from_roll_list
from rhythmic_relationships.parts import PARTS
from rhythmic_relationships.representations import REPRESENTATIONS

dataset_name = "lmdc_1000_1bar_4res"
representation = "roll"
n_parts_min = 2

# Load dataset annotations
df = load_dataset_annotations(dataset_name)

# Sort instruments by part index for convenience
df["part_ix"] = df["part_id"].apply(lambda x: PARTS.index(x)).values
df.sort_values("part_ix", inplace=True)

# Choose a random segment with more than `n_parts_min` parts
n_parts_by_seg = df.groupby(["segment_id", "filepath"]).apply(len)

for n in range(5):
    seg_id, fp = np.random.choice(n_parts_by_seg[n_parts_by_seg >= n_parts_min].index)
    seg_df = df.loc[(df.filepath == fp) & (df.segment_id == seg_id)]

    seg_reprs = []
    for i, row in seg_df.iterrows():
        npz = np.load(fp, allow_pickle=True)
        # TODO: Handle multiple rolls from the same part. For now we just take the first one
        reprs = npz[f"{row['segment_id']}_{row['part_id']}"][0]
        seg_reprs.append(reprs[REPRESENTATIONS.index(representation)])

    parts = seg_df.part_id.tolist()
    filename = os.path.splitext(os.path.basename(fp))[0]
    print(f"Selected segment {seg_id} from {filename}\n\tParts: {parts}\n")
    write_midi_from_roll_list(seg_reprs, f"{filename}_{seg_id}.mid", binary=True, parts=parts)
