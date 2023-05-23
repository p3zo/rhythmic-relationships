import os

from torch.utils.data import DataLoader
from rhythmic_relationships import DATASETS_DIR
from rhythmic_relationships.data import PartPairDataset
from rhythmic_relationships.io import write_midi_from_roll

dataset_name = "lmdc_3000_2bar_4res"
part_1 = "Melody"
part_2 = "Bass"
n_samples = 16

samples_dir = os.path.join(DATASETS_DIR, dataset_name, "samples")
if not os.path.isdir(samples_dir):
    os.makedirs(samples_dir)

dataset = PartPairDataset(
    dataset_name=dataset_name,
    part_1=part_1,
    part_2=part_2,
    repr_1="onset_roll",
    repr_2="onset_roll",
)
loader = DataLoader(dataset, batch_size=n_samples, shuffle=True)

x = next(iter(loader))

src, tgt = x

for ix, roll in enumerate(x):
    seg_id = dataset.loaded_segment_ids[ix].replace("/", "_")
    outpath = os.path.join(samples_dir, f"{seg_id}.mid")
    write_midi_from_roll(
        roll.numpy(),
        outpath=outpath,
        part=part,
        binary=False,
        onset_roll=True,
    )
    print(f"Wrote {outpath}")
