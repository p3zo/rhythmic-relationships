import glob
import os

from rhythmic_relationships import DATASETS_DIR
from rhythmic_relationships.io import load_midi_file, slice_midi


def get_interonset_interval(hits):

    return

seg_size = 2
resolution = 4
n_beat_bars = 4
min_seg_pitches = 0
min_seg_beats = 0
min_melody_pitches = 0
max_melody_rests = 64
representations = [
    "onset_roll",
    "drum_roll",
    "hits",
    "drum_hits",
    "descriptors",
]
mv_samples_dir = os.path.join(DATASETS_DIR, "baseline", "music-vae-4bar-trios-samples")
mv_samples_paths = glob.glob(f"{mv_samples_dir}/*.mid")
# fp = mv_samples_paths[0]
# fp = os.path.join(DATASETS_DIR, "baseline", "trio.mid")
print(f"{len(mv_samples_paths)=}")
for fp in mv_samples_paths:
    pmid = load_midi_file(fp)
    for i in pmid.instruments:
        if i.name == "Drums":
            i.is_drum = True
    seg_part_reprs = slice_midi(
        pmid=pmid,
        seg_size=seg_size,
        resolution=resolution,
        n_beat_bars=n_beat_bars,
        min_seg_pitches=min_seg_pitches,
        min_seg_beats=min_seg_beats,
        min_melody_pitches=min_melody_pitches,
        max_melody_rests=max_melody_rests,
        representations=representations,
    )
    if not list(seg_part_reprs.keys()) == [
        "0_Bass",
        "1_Bass",
        "0_Melody",
        "1_Melody",
        "0_Drums",
        "1_Drums",
    ]:
        print(f"{fp=}")


print("stop")
print("here")
