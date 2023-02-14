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

### Create the dataset

Slice MIDI data into segments and aggregate the segments by part using `scripts/prepare_data.py`. It accepts either a
MIDI file or a directory of MIDI files. For reference, it took ~1.5hrs to process
the [LMD clean subset](https://colinraffel.com/projects/lmd/) (17243 MIDI files).

    python prepare_data.py --path=input/slakh00006/all_src.mid --prefix=slakh00006

For each MIDI file, an `.npz` file is written containing [piano roll](https://en.wikipedia.org/wiki/Piano_roll#In_digital_audio_workstations) representations of the segments
in that file organized by part. The piano rolls are arrays of type `numpy.uint8` and shape `(S x N x V)`, where `S` is
the number of segments, `N` is the number of time steps in a segment, and `V` is the number of MIDI pitches. The array
values are MIDI velocities in the range `[0-127]`. Additionally, all segments for each part are collected into a single
`part_segrolls` file containing one `(S x N x V)` matrix.

The piano roll images created with the `--create_images` flag can be listened to using the notebook
[MIDI_playback_from_image.ipynb](MIDI_playback_from_image.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1okATUg3TI1CsyKi1OUsQTt8FB28XfIm1?usp=sharing).
