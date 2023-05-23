"""Load a single MIDI segment from a dataset."""
import argparse
import os

from rhythmic_relationships import DATASETS_DIR
from rhythmic_relationships.data import (
    load_dataset_annotations,
    get_seg_fname,
)
from rhythmic_relationships.io import (
    get_pmid_segment,
    get_pmid_segment_reprs,
    load_midi_file,
    write_midi_from_hits,
)
from rhythmic_relationships.parts import PARTS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="lmdc_250_1bar_4res",
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
        default="Ron/Sono uguale a te",
        help="The name of the MIDI file to load from.",
    )
    parser.add_argument(
        "--segment_id",
        type=int,
        default=107,
        help="The ID of the segment to load.",
    )
    parser.add_argument(
        "--parts",
        nargs="+",
        default=[],
        help="A list of parts to include in the segment. If empty, all parts will be included.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/midi",
        help="The directory to save the output MIDI files to.",
    )

    args = parser.parse_args()

    dataset_name = args.dataset
    seg_id = args.segment_id
    filename = args.filename
    midi_dir = args.midi_dir
    output_dir = args.output_dir
    parts = args.parts

    file_seg_id = f"{filename.replace('/', '_')}_{seg_id}"

    # Load dataset annotations
    df = load_dataset_annotations(os.path.join(DATASETS_DIR, dataset_name))
    df["filename"] = df["filepath"].apply(
        lambda x: get_seg_fname(x, os.path.join(DATASETS_DIR, dataset_name))
    )

    seg_df = df.loc[(df.filename == filename) & (df.segment_id == seg_id)]

    pmid = load_midi_file(os.path.join(midi_dir, f"{filename}.mid"))

    if not parts:
        parts = seg_df.part_id.values.tolist()
        # Sort by part index for convenience
        parts.sort(key=lambda x: PARTS.index(x))

    # Slice first way
    pmid_roll, pmid_onset_roll, pmid_onset_roll_3o, hits_list = get_pmid_segment_reprs(
        pmid, seg_id, parts
    )
    outpath = os.path.join(output_dir, f"{file_seg_id}_roll.mid")
    pmid_roll.write(outpath)
    print(f"Wrote {outpath}")

    outpath_or = os.path.join(output_dir, f"{file_seg_id}_or.mid")
    pmid_onset_roll.write(outpath_or)
    print(f"Wrote {outpath_or}")

    outpath_or3 = os.path.join(output_dir, f"{file_seg_id}_or3.mid")
    pmid_onset_roll_3o.write(outpath_or3)
    print(f"Wrote {outpath_or3}")

    for (part, hits) in zip(parts, hits_list):
        hits_outpath = os.path.join(output_dir, f"{file_seg_id}_{part}.mid")
        write_midi_from_hits(hits, hits_outpath)
        print(f"Wrote {hits_outpath}")

    # Slice second way
    # TODO: debug get_pmid_segment
    pmid_slice_2 = get_pmid_segment(pmid, segment_num=seg_id, parts=parts)

    outpath = os.path.join(output_dir, f"{file_seg_id}_slice2.mid")
    pmid_slice_2.write(outpath)
    print(f"Wrote {outpath}")
