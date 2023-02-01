# Rhythmic complements

Use `prepare_data.py` to create the dataset. It accepts either a MIDI file or a directory of MIDI files.

Example

    python prepare_data.py --path=input/slakh00006/all_src.mid --prefix=slakh0006

Instructions

    usage: prepare_data.py [-h] --path PATH [--seg_size SEG_SIZE] [--resolution RESOLUTION] [--prefix PREFIX] [--drum_roll] [--create_images] [--image_size IMAGE_SIZE]
                      [--compute_descriptors] [--pypianoroll_plots] [--verbose]

    options:
      -h, --help            show this help message and exit
      --path PATH           Path to the input: either a MIDI file or a directory of MIDI files.
      --seg_size SEG_SIZE   Number of bars per segment.
      --resolution RESOLUTION
                            Number of subdivisions per beat.
      --prefix PREFIX       An identifier for output filenames.
      --drum_roll           Use a 9-voice piano roll for drums only.
      --create_images       Create images of the piano rolls.
      --image_size IMAGE_SIZE
                            A resolution to use for images, e.g. 512x512.
      --compute_descriptors
                            Use rhythmtoolbox to compute rhythmic descriptors for each segment.
      --pypianoroll_plots   Create a pypianoroll plot for each segment and another for the entire track.
      --verbose             Print debug statements.


Use the [MIDI_playback_from_image](https://colab.research.google.com/drive/1okATUg3TI1CsyKi1OUsQTt8FB28XfIm1?usp=sharing) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1okATUg3TI1CsyKi1OUsQTt8FB28XfIm1?usp=sharing)
notebook to listen to piano roll images.
