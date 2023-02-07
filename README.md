# Rhythmic complements

## Usage

### Recreate the dataset

#### 1. Process MIDI

Run `prepare_data.py` to split the data into segments and aggregate parts. It accepts either a MIDI file or a directory
of MIDI files. For reference, it took ~1hr 20m to process
the [LMD clean subset](https://colinraffel.com/projects/lmd/) (17243 MIDI files).

    python prepare_data.py --path=input/slakh00006/all_src.mid --prefix=slakh00006

Full list of arguments

    usage: prepare_data.py [-h] [--path PATH] [--seg_size SEG_SIZE] [--resolution RESOLUTION] [--prefix PREFIX] [--drum_roll] [--create_images]
                           [--im_size IM_SIZE] [--compute_descriptors] [--pypianoroll_plots] [--verbose]

    options:
      -h, --help               show this help message and exit
      --path PATH              Path to the input: either a MIDI file or a directory of MIDI files.
      --seg_size SEG_SIZE      Number of bars per segment.
      --resolution RESOLUTION  Number of subdivisions per beat.
      --prefix PREFIX          An identifier for output filenames.
      --drum_roll              Use a 9-voice piano roll for drums only.
      --create_images          Create images of the piano rolls.
      --im_size IM_SIZE        A resolution to use for the piano roll images, e.g. 512x512.
      --compute_descriptors    Use rhythmtoolbox to compute rhythmic descriptors for each segment.
      --pypianoroll_plots      Create a pypianoroll plot for each segment and another for the entire track.
      --verbose                Print debug statements.

One `.npz` file is written for each MIDI file and contains a `numpy.uint8` array for each part of shape `(S x N x V)`,
where `S` is the number of segments, `N` is the number of time steps in a segment, and `V` is the number of voices. The
values are MIDI velocities in the range `[0-127]`.

The piano roll images created with the `--create_images` flag can be listened to using the notebook
[MIDI_playback_from_image.ipynb](MIDI_playback_from_image.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1okATUg3TI1CsyKi1OUsQTt8FB28XfIm1?usp=sharing).
