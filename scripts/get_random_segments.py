"""Select segments with at least n parts from a dataset at random."""

import argparse
import os

import numpy as np
from rhythmic_relationships import DATASETS_DIR
from rhythmic_relationships.data import load_dataset_annotations
from rhythmic_relationships.io import get_pmid_segment_reprs, load_midi_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="lmdc_1000_2bar_4res",
        help="Name of the dataset to make plots for.",
    )
    parser.add_argument(
        "--midi_dir",
        type=str,
        default="../input/lmd_clean",
        help="Name of the directory from which to load MIDI data.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="Cocciante/Cervo a primavera",
        help="The name of the MIDI file to load from.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="The number of samples to select.",
    )
    parser.add_argument(
        "--n_parts_min",
        type=int,
        default=2,
        help="The minimum number of parts to include in a segment.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/midi",
        help="The directory to save the output MIDI files to.",
    )

    args = parser.parse_args()

    dataset_name = args.dataset
    filename = args.filename
    midi_dir = args.midi_dir
    n_samples = args.n_samples
    n_parts_min = args.n_parts_min
    output_dir = args.output_dir

    # Load dataset annotations
    df = load_dataset_annotations(os.path.join(DATASETS_DIR, dataset_name))
    df = df.sample(frac=1)  # shuffle

    # Choose a random segment with more than `n_parts_min` parts
    n_parts_by_seg = df.groupby(["segment_id", "filepath"]).apply(len)

    for n in range(n_samples):
        seg_id, fp = np.random.choice(
            n_parts_by_seg[n_parts_by_seg >= n_parts_min].index
        )
        seg_df = df.loc[(df.filepath == fp) & (df.segment_id == seg_id)]

        track_name = os.path.splitext("/".join(fp.split("/")[-2:]))[0]
        pmid = load_midi_file(os.path.join(midi_dir, f"{track_name}.mid"))

        parts = seg_df.part_id.values.tolist()
        (
            pmid_roll,
            pmid_onset_roll,
            pmid_onset_roll_3o,
            hits_list,
        ) = get_pmid_segment_reprs(pmid, seg_id, parts=parts)

        file_seg_id = f"{os.path.splitext(os.path.basename(fp))[0]}_{seg_id}"
        print(f"Selected segment {file_seg_id}\n\tParts: {parts}\n")

        pmid_onset_roll.write(os.path.join(output_dir, f"{file_seg_id}.mid"))
