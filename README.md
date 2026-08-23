# Usage

## Environment

This is a [uv](https://docs.astral.sh/uv/) project. To create `.venv` in the repo and install
everything into it:

```bash
uv sync --extra experimental
```

Versions come from the committed `uv.lock`, so every checkout resolves identically. `rhythmtoolbox`
is pulled from git. The `experimental` extra is only needed by the analysis scripts; the test
dependencies are in the default `dev` group, so they are always installed.

Run anything in the environment with `uv run`, which re-syncs first if the lockfile has changed:

```bash
uv run pytest
uv run python scripts/modeling/run_train.py --model hits_encdec
```

Change dependencies with `uv add` / `uv remove` rather than by hand, so the lockfile stays in step.

## Play with a trained model

Two ways, the same interface.

**Hosted**, at [p3zo.github.io/rhythmic-relationships](https://p3zo.github.io/rhythmic-relationships/) — the model runs in the browser through onnxruntime-web, so there is nothing to install. It ships one exported model and 30,000 indexed segments to search.

**Locally**, against any model you have trained, with the full rhythmtoolbox descriptors:

```bash
uv run python scripts/serve_model.py     # http://localhost:8765
```

To refresh the hosted site after training a new model, re-export and commit `docs/`:

```bash
uv run python scripts/export_web.py --model_path output/models/hits_encdec/<pair>/<run>/model.pt
```

The export cannot run in CI, since it needs both the checkpoint and the prepared dataset, so
`.github/workflows/pages.yml` only publishes what is committed. The JavaScript reimplements the
sampling loop, the hits vocabulary and the paired descriptors; `scripts/check_web_port.py` plus
`scripts/check_web_port.mjs` check the browser against the Python for all three.

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

### Get the MIDI data

`input/babyslakh` ships with the repo: 20 tracks from [BabySlakh](https://zenodo.org/record/4603870), enough to exercise the code but too small to train on.

Every config under `scripts/modeling/` names `lmdc_17243_2bar_4res`, which is built from the `clean_midi` subset of the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) — 223 MB to download, 811 MB extracted:

```bash
curl -L -o input/clean_midi.tar.gz http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz
tar -xzf input/clean_midi.tar.gz -C input/
rm input/clean_midi.tar.gz
```

`input/` is gitignored. The subset holds 17,256 `.mid` files, but 13 of them sit under a dot-prefixed directory that `glob` skips, which is where the `17243` in the dataset name comes from.

### Create a dataset

Slice MIDI data into segments and aggregate the segments by part using `scripts/prepare_dataset.py`. It accepts either a MIDI file or a directory of MIDI files. To process the example input:

    uv run python scripts/prepare_dataset.py --path=input/babyslakh --prefix=babyslakh --seg_size=1

To build the dataset the model configs expect, which takes about an hour:

    uv run python scripts/prepare_dataset.py --path=input/clean_midi --prefix=lmdc --seg_size=2 --resolution=4

Datasets are named `{prefix}_{n_midi_files}_{seg_size}bar_{resolution}res`, and that name is what a config's `data.dataset_name` refers to. A config's `sequence_len` has to equal `seg_size * n_beat_bars * resolution`; nothing checks the two against each other, so a mismatch surfaces as a shape error inside the model.

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
