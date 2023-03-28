"""
Mashup segments using rhythmic relationships.
Stitch the drums from one segment with the bassline of a second and the melody from a third.
"""
import numpy as np
from rhythmic_relationships.data import PartPairDataset
from rhythmic_relationships.io import write_midi_from_roll_list
from torch.utils.data import DataLoader

dataset_name = "musicnet_330_2bar_4res"
part_1 = "Piano"
part_2 = "Orchestra Solo"
bass_piano_dataset = PartPairDataset(
    **{
        "dataset_name": dataset_name,
        "part_1": part_1,
        "part_2": part_2,
        "repr_1": "roll",
        "repr_2": "roll",
    }
)

batch_size = 2

n_measures = 100
bass_rolls = []
piano_rolls = []

for n in range(n_measures):
    bass_piano_loader = DataLoader(
        bass_piano_dataset, batch_size=batch_size, shuffle=True
    )

    bass, piano = next(iter(bass_piano_loader))

    # Pair a bass with a piano from a different segment
    bass_rolls.append(bass[0])
    piano_rolls.append(piano[1])

bass_roll = np.concatenate(bass_rolls)
piano_roll = np.concatenate(piano_rolls)

write_midi_from_roll_list(
    [bass_roll, piano_roll],
    f"output/{dataset_name}_{part_1}_{part_2}_{n_measures}.mid",
    binary=True,
    parts=["Bass", "Piano"],
)
