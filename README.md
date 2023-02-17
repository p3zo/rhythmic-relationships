# Usage

## Environment

- Install [PyTorch](https://pytorch.org/get-started/locally/) according to your OS and GPU.
- Install this repo and [rhythmtoolbox](https://github.com/danielgomezmarin/rhythmtoolbox) from source:

```
pip install git+https://github.com/p3zo/rhythmic-complements git+https://github.com/danielgomezmarin/rhythmtoolbox
```

## Inference

To generate a bass pattern given a drum roll, use `scripts/train_0a.py`

## Training

To train a conditional VAE for a pair of Bass patterns and Drum rolls, use `scripts/train_0a.py`

## Dataset

### Load the dataset

Torch `Dataset` classes are provided for both unconditional and conditional use-cases. Each class allows for flexibility
between three types of representations:

1. `roll`: a [piano roll](https://en.wikipedia.org/wiki/Piano_roll#In_digital_audio_workstations) matrix
2. `pattern`: a vector of monophonic onset times
3. `descriptor`: a vector of rhythmic descriptors computed
   using [rhythmtoolbox](https://github.com/danielgomezmarin/rhythmtoolbox)

A dataset of segment pairs can be loaded via `PairDataset`. An example of loading an (X, y) dataset of `Bass` rolls
paired with `Drums` patterns:

```python
from rhythmic_complements.data import PairDataset
from torch.utils.data import DataLoader

dataset_dir = 'path/to/your/dataset'
dataset = PairDataset(dataset_dir, 'Bass', 'Drums', "roll", "pattern")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

x, y = next(iter(loader))
print(f"x batch shape: {x.size()}")
print(f"y batch shape: {y.size()}")
```

### Create the dataset

Slice MIDI data into segments and aggregate the segments by part using `scripts/prepare_data.py`. It accepts either a
MIDI file or a directory of MIDI files. For reference, it took ~1.5 hours to process
the [LMD clean subset](https://colinraffel.com/projects/lmd/) (17243 MIDI files) and the size of the resulting dataset
was ~3 GB. To process the example input provided from the [BabySlakh](https://zenodo.org/record/4603870) dataset:

    python scripts/prepare_data.py --path=input/babyslakh --prefix=babyslakh --seg_size=1 --binarize

For each MIDI file, an `.npz` file is written containing piano roll representations of the segments in that file
organized by segment and part. The piano rolls are arrays of type `numpy.uint8` and shape `(S x N x V)`, where `S` is
the number of segments, `N` is the number of time steps in a segment, and `V` is the number of MIDI pitches. If
the `--binarize` option is used, the array values are either `0` or `1` representing onsets, else they are integer MIDI
velocities in the range `[0-127]`. A map of all the segments is saved in the top-level directory as `rolls.csv`.
Additionally, a set of lookup tables for all segment pairs are stored in the `pair_lookups`directory, one for each pair
of parts. Finally, two plots displaying the distribution of segments by both part and part pair are saved to the
top-level directory. The final dataset directory structure looks like this, assuming 3 parts and a flat input directory
of MIDI files:

```
├── pair_lookups
│   ├── Part1_Part2.csv
│   ├── Part1_Part3.csv
│   ├── Part2_Part3.csv
├── rolls
│   ├── track1.npz
│   ├── track2.npz
│   └── track3.npz
├── rolls.csv
├── segments_by_part.png
└── segments_by_part_pair.png
```

The piano roll images created with the `--create_images` flag can be listened to using the notebook
[MIDI_playback_from_image.ipynb](MIDI_playback_from_image.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1okATUg3TI1CsyKi1OUsQTt8FB28XfIm1?usp=sharing).
