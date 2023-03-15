# Usage

## Environment

- Install [PyTorch](https://pytorch.org/get-started/locally/) according to your OS and GPU.
- Install this repo and [rhythmtoolbox](https://github.com/danielgomezmarin/rhythmtoolbox) from source:

```
pip install git+https://github.com/p3zo/rhythmic-relationships git+https://github.com/danielgomezmarin/rhythmtoolbox
```

## Inference

To generate a bass pattern given a drum roll, use `scripts/train_0a.py`

## Training

To train a conditional VAE for a pair of Bass patterns and Drum rolls, use `scripts/train_0a.py`

## Data

### Representations

Rhythms can be encoded into various representations, each of which gives a different view into rhythmic relationships.
This repository implements the following:

- `roll`: a [piano roll](https://en.wikipedia.org/wiki/Piano_roll#In_digital_audio_workstations) with MIDI velocities
  converted to real numbers in [0, 1]
- `chroma`: a ternary [chroma](https://en.wikipedia.org/wiki/Chroma_feature). `0` is a silence, `1` is an onset, and `2`
  is a continuation of a previous onset.
- `pattern`: a ternary vector of onsets and offsets. `0` is a silence, `1` is an onset, and `2` is a continuation of a
  previous onset. Onsets take precedence over continuations.
- `hits`: a vector of onsets with MIDI velocities converted to real numbers in [0, 1]
- `descriptors`: a vector of rhythmic descriptors computed
  using [rhythmtoolbox](https://github.com/danielgomezmarin/rhythmtoolbox)

### Load a dataset

A dataset of segment pairs can be loaded via the `PairDataset` class, which is a Torch `Dataset` that takes parameters
to allow for flexibility between parts and representations. For example, to load a dataset of `Bass` patterns paired
with `Drums` hits:

```python
from rhythmic_relationships.data import PairDataset
from torch.utils.data import DataLoader

dataset_config = {
    "dataset_name": "babyslakh_20_1bar_4res",
    "part_1": "Bass",
    "part_2": "Drums",
    "repr_1": "pattern",
    "repr_2": "hits",
}
dataset = PairDataset(**dataset_config)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

x, y = next(iter(loader))
print(f"x batch shape: {x.size()}")
print(f"y batch shape: {y.size()}")
```

### Create a dataset

Slice MIDI data into segments and aggregate the segments by part using `scripts/prepare_dataset.py`. It accepts either a
MIDI file or a directory of MIDI files. To process the example input provided from
the [BabySlakh](https://zenodo.org/record/4603870) dataset:

    python scripts/prepare_dataset.py --path=input/babyslakh --prefix=babyslakh --seg_size=1 --binarize

For each MIDI file, an `.npz` file is written containing piano roll representations of the segments in that file
organized by segment and part. The piano rolls are arrays of type `numpy.uint8` and shape `(S x N x V)`, where `S` is
the number of segments, `N` is the number of time steps in a segment, and `V` is the number of MIDI pitches. If
the `--binarize` option is used, the array values are either `0` or `1` representing onsets, else they are integer MIDI
velocities in the range `[0-127]`. A map of all the segments is saved in the top-level directory as `segments.csv`.
Additionally, a set of lookup tables for all segment pairs are stored in the `pair_lookups`directory, one for each pair
of parts. Finally, two plots displaying the distribution of segments by both part and part pair are saved to the `plots`
directory. The final dataset directory structure looks like this, assuming 3 parts and a flat input directory of MIDI
files:

```
├── pair_lookups
│   ├── part1_part2.csv
│   ├── part1_part3.csv
│   ├── part2_part3.csv
├── representations
│   ├── track1.npz
│   ├── track2.npz
│   └── track3.npz
├── plots
│   ├── segments_by_part.png
│   └── segments_by_part_pair.png
└── segments.csv
```

The piano roll images created with the `--create_images` flag can be listened to using the notebook
[MIDI_playback_from_image.ipynb](scripts/MIDI_playback_from_image.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1okATUg3TI1CsyKi1OUsQTt8FB28XfIm1?usp=sharing).
