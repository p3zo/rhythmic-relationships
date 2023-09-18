# Usage

## Environment

- Install [PyTorch](https://pytorch.org/get-started/locally/) according to your OS and GPU.
- Install this repo. Remove the `[experimental]` suffix if you aren't planning to do any development.

```bash
pip install -e .[experimental]
```

- Install [rhythmtoolbox](https://github.com/danielgomezmarin/rhythmtoolbox) from source.

```
pip install git+https://github.com/danielgomezmarin/rhythmtoolbox
```

## Scripts

See [scripts/README.md](scripts/README.md) for a list of scripts and notebooks.

## Data

### Parts

The "parts" of a piece of music are the strands of melody or harmony that make up the musical texture. The grouping of instruments into parts depends on musical style. For styles like pop, rock, or jazz, an appropriate set of parts might be percussive, bassline, harmonic, and melodic. We adopt the approach of MusicVAE [Roberts et al, 2018] which defines three parts to model multi-stream music: Drums, Bass, and Melody. Because there is no publicly available dataset with these category labels, we use the instrumental categories from the [General MIDI spec (Level 2)](https://en.wikipedia.org/wiki/General_MIDI_Level_2).

### Representations

Rhythms can be encoded into various representations, each of which gives a different view into rhythmic relationships. This repository implements the following:

- `roll`: a [piano roll](https://en.wikipedia.org/wiki/Piano_roll#In_digital_audio_workstations) with MIDI velocities converted to real numbers in [0, 1]
- `onset_roll`: a `roll` with only onsets
- `onset_roll_3_octave`: an `onset_roll` with pitches mapped to three octaves centered around C4 with range [48, 84]
- `binary_onset_roll`: an `onset_roll` with `0` or `1` representing onsets
- `drum_roll`: a `roll` with pitches mapped to 9 drum voices following the "Paper Mapping" of the [Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove)
- `chroma`: a ternary [chroma](https://en.wikipedia.org/wiki/Chroma_feature). `0` is a silence, `1` is an onset, and `2`
  is a continuation of a previous onset.
- `pattern`: a ternary vector of onsets and offsets. `0` is a silence, `1` is an onset, and `2` is a continuation of a previous onset. Onsets take precedence over continuations.
- `hits`: a vector of onsets with MIDI velocities converted to real numbers in [0, 1]
- `descriptors`: a vector of rhythmic descriptors computed using [rhythmtoolbox](https://github.com/danielgomezmarin/rhythmtoolbox)

### Load a dataset

We use Torch `Dataset` classes to allow for flexibility to load datasets with different parts and representations.

A dataset of part segments can be loaded via the `PartDataset` class. For example, to load a dataset of `Guitar` rolls:

```python
from rhythmic_relationships.data import PartDataset
from torch.utils.data import DataLoader

dataset_config = {
    "dataset_name": "babyslakh_20_1bar_4res",
    "part": "Guitar",
    "representation": "roll",
}
dataset = PartDataset(**dataset_config)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

x = next(iter(loader))
print(f"x batch shape: {x.size()}")
```

A dataset of segment pairs can be loaded via the `PartPairDataset` class. For example, to load a dataset of `Bass`
patterns paired with `Drums` hits:

```python
from rhythmic_relationships.data import PartPairDataset
from torch.utils.data import DataLoader

dataset_config = {
    "dataset_name": "babyslakh_20_1bar_4res",
    "part_1": "Bass",
    "part_2": "Drums",
    "repr_1": "pattern",
    "repr_2": "hits",
}
dataset = PartPairDataset(**dataset_config)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

x, y = next(iter(loader))
print(f"x batch shape: {x.size()}")
print(f"y batch shape: {y.size()}")
```

### Create a dataset

Slice MIDI data into segments and aggregate the segments by part using `scripts/prepare_dataset.py`. It accepts either a MIDI file or a directory of MIDI files. To process the example input provided from the [BabySlakh](https://zenodo.org/record/4603870) dataset:

    python scripts/prepare_dataset.py --path=input/babyslakh --prefix=babyslakh --seg_size=1

One `.npz` file is created for each MIDI file in the dataset. Each `.npz` file contains the representations of the segments of its corresponding MIDI file. The representations are stored in arrays of type `numpy.uint8` and shape `(S x N x V)`, where `S` is the number of segments, `N` is the number of time steps in a segment, and `V` is the number of MIDI pitches. A map of all the segments is saved in the top-level directory as `segments.csv`. Lookup tables for co-occurring segments are stored in the `pair_lookups` directory, one for each pair of parts. Plots displaying dataset distributions are saved to the `plots` directory. An example dataset directory is shown below:

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
│   ├── plot1.png
│   └── plot2.png
└── segments.csv
```

## Models

To train a model:

    python scripts/modeling/run_train.py --model hits_encdec

The possible model names are the names of directories in `scripts/modeling`. See `scripts/modeling/README.md` for a catalog.
