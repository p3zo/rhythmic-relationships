import os

from rhythmic_relationships import DATASETS_DIR
from rhythmic_relationships.data import PartPairDataset, get_seg_fname
from rhythmic_relationships.io import write_midi_from_roll
from torch.utils.data import DataLoader


def get_random_part_pairs(dataset_name, part_1, part_2, n_samples):
    dataset = PartPairDataset(
        dataset_name=dataset_name,
        part_1=part_1,
        part_2=part_2,
        repr_1="drum_roll" if part_1 == "Drums" else "onset_roll",
        repr_2="drum_roll" if part_2 == "Drums" else "onset_roll",
    )
    loader = DataLoader(dataset, batch_size=n_samples, shuffle=True)

    batch = next(iter(loader))

    return batch, dataset.loaded_segments


dataset_name = "lmdc_17243_2bar_4res"
part_1 = "Melody"
part_2 = "Bass"
n_samples = 25
batch, loaded_segments = get_random_part_pairs(dataset_name, part_1, part_2, n_samples)
srcs, tgts = batch

samples_dir = os.path.join(DATASETS_DIR, dataset_name, "samples")
if not os.path.isdir(samples_dir):
    os.makedirs(samples_dir)

for ix in range(n_samples):
    src = srcs[ix]
    tgt = tgts[ix]

    fp, sid = loaded_segments[ix]
    seg_name = get_seg_fname(fp, os.path.join(DATASETS_DIR, dataset_name))

    src_outpath = os.path.join(
        samples_dir, f"{seg_name.replace('/', '_')}_{part_1}.mid"
    )
    write_midi_from_roll(
        src.numpy(),
        outpath=src_outpath,
        part=part_1,
        scaled=True,
        binary=False,
        onset_roll=True,
    )

    tgt_outpath = os.path.join(
        samples_dir, f"{seg_name.replace('/', '_')}_{part_2}.mid"
    )
    write_midi_from_roll(
        tgt.numpy(),
        outpath=tgt_outpath,
        part=part_2,
        scaled=True,
        binary=False,
        onset_roll=True,
    )
    print(f"Wrote {src_outpath} and {tgt_outpath}")
