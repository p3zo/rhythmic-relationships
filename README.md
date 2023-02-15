# Rhythmic complements

## Usage

### Environment

- Install [PyTorch](https://pytorch.org/) according to your OS and GPU.
- Install this repo and [rhythmtoolbox](https://github.com/danielgomezmarin/rhythmtoolbox) from source

```
pip install git+https://github.com/p3zo/rhythmic-complements git+https://github.com/danielgomezmarin/rhythmtoolbox
```

### Inference

To generate a bass part given a random drum part from the dataset, run `scripts/predict.py`.

### Training

To train a conditional VAE for a pair of rhythm space points (RSPs), use `scripts/rsp_pairs_train.py`.

To train a VAE for a single part using a rhythm descriptor representation, use `scripts/rsp_train.py`. Example
inference is given in `scripts/rsp_inference.py`.

To train a VAE for a single part using a piano roll representation, use `scripts/segroll_train.py`. Example inference
is given in `scripts/segroll_inference.py`.

### Dataset

#### Load a dataset

Torch `Dataset` classes are provided for both unconditional and conditional use-cases. Each class allows for flexibility
between two types of representions: `roll` which refers to the piano roll representation, and `desc` which refers to the
rhythmic descriptor representation.

A dataset of segment pairs can be loaded via `PairDataset`. An example of loading an (X, y) dataset of `Drums` rolls
paired with `Guitar` descriptors:

```python
from rhythmic_complements.data import PairDataset
from torch.utils.data import DataLoader

dataset = PairDataset(DATASET_DIR, 'Drums', 'Guitar', "roll", "desc")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

x, y = next(iter(loader))
print(f"x batch shape: {x.size()}")
print(f"y batch shape: {y.size()}")
```

#### Create a new dataset

Slice MIDI data into segments and aggregate the segments by part using `scripts/prepare_data.py`. It accepts either a
MIDI file or a directory of MIDI files. For reference, it took ~1.5hrs to process
the [LMD clean subset](https://colinraffel.com/projects/lmd/) (17243 MIDI files).

    python prepare_data.py --path=input/slakh00006/all_src.mid --prefix=slakh00006 --seg_size=1 --binarize --subset=10

For each MIDI file, an `.npz` file is written
containing [piano roll](https://en.wikipedia.org/wiki/Piano_roll#In_digital_audio_workstations) representations of the
segments in that file organized by segment and part. The piano rolls are arrays of type `numpy.uint8` and
shape `(S x N x V)`, where `S` is the number of segments, `N` is the number of time steps in a segment, and `V` is the
number of MIDI pitches. If the `--binarize` option is used, the array values are either `0` or `1` representing onsets,
else they are integer MIDI velocities in the range `[0-127]`. A map of all the segments is saved in the top-level
directory as `rolls.csv`. Additionally, a set of lookup tables for all segment pairs are stored in the `pair_lookups`
directory, one for each pair of parts. Finally, two plots displaying the distribution of segments by both part and part
pair are saved to the top-level directory. The final dataset directory structure looks like this, assuming 3 parts and a
flat input directory of MIDI files:

```
├── pair_lookups
│   ├── Drums_Bass.csv
│   ├── Drums_Guitar.csv
│   ├── Guitar_Bass.csv
├── rolls
│   ├── track1.npz
│   ├── track2.npz
│   └── track3.npz
├── rolls.csv
├── segments_by_part.png
└── segments_by_part_pair.png
```

If the `--compute_descriptors` flag is used, an additional file will be written to the top-level directory containing
the descriptors for all segments.

The piano roll images created with the `--create_images` flag can be listened to using the notebook
[MIDI_playback_from_image.ipynb](MIDI_playback_from_image.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1okATUg3TI1CsyKi1OUsQTt8FB28XfIm1?usp=sharing).
