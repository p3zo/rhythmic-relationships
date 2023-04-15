"""Load a single MIDI segment from a dataset."""
import argparse
import os

from rhythmic_relationships import DATASETS_DIR, REPRESENTATIONS_DIRNAME
from rhythmic_relationships.data import (
    load_dataset_annotations,
    split_filepath_on_dirname,
)
from rhythmic_relationships.io import (
    get_pmid_segment,
    load_midi_file,
    slice_midi,
    get_pretty_midi_from_roll_list,
    write_midi_from_hits,
)
from rhythmic_relationships.parts import PARTS
from rhythmic_relationships.representations import REPRESENTATIONS


def slice_first_way(pmid, segment_id, parts):
    part_seg_reprs = slice_midi(pmid)

    hits_list = []
    roll_list = []
    for part in parts:
        reprs = part_seg_reprs[f"{segment_id}_{part}"][0]
        roll_list.append(reprs[REPRESENTATIONS.index("roll")] * 127)

        hits_list.append(reprs[REPRESENTATIONS.index("hits")] * 127)

    return get_pretty_midi_from_roll_list(roll_list, parts=parts), hits_list


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
        default="Cocciante/Cervo a primavera",
        help="The name of the MIDI file to load from.",
    )
    parser.add_argument(
        "--segment_id",
        type=int,
        default=130,
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
        lambda x: split_filepath_on_dirname(
            x, os.path.join(DATASETS_DIR, dataset_name, REPRESENTATIONS_DIRNAME)
        )
    )

    seg_df = df.loc[(df.filename == filename) & (df.segment_id == seg_id)]

    pmid = load_midi_file(os.path.join(midi_dir, f"{filename}.mid"))

    if not parts:
        parts = seg_df.part_id.values.tolist()
        # Sort by part index for convenience
        parts.sort(key=lambda x: PARTS.index(x))

    # Slice first way
    pmid_slice_1, hits_list = slice_first_way(pmid, seg_id, parts)
    outpath = os.path.join(output_dir, f"{file_seg_id}_roll.mid")
    pmid_slice_1.write(outpath)
    print(f"Wrote {outpath}")

    for (part, hits) in zip(parts, hits_list):
        hits_outpath = os.path.join(output_dir, f"{file_seg_id}_{part}.mid")
        write_midi_from_hits(hits, hits_outpath)
        print(f"Wrote {hits_outpath}")


    # Slice second way
    # TODO: second way is not working yet
    pmid_slice_2 = get_pmid_segment(pmid, segment_num=seg_id, parts=parts)

    outpath = os.path.join(output_dir, f"{file_seg_id}.mid")
    pmid_slice_2.write(outpath)
    print(f"Wrote {outpath}")
