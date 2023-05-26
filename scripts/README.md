## Scripts

### Data

`prepare_dataset.py`: Create a new dataset. See instructions in [Create the dataset](../README.md#create-the-dataset)

`create_chroma_vocabulary.ipynb`: A notebook that creates a reduced vocabulary of chord types to use for modeling chromas.

`MIDI_playback_from_image.ipynb`: Piano roll images can be synthesized to audio to using this notebook. It is also available in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1okATUg3TI1CsyKi1OUsQTt8FB28XfIm1?usp=sharing).

### Analysis

`pairspace.py`: performs dimensionality reduction over PartPairDatasets using both MDS and t-SNE, and visualizes the resulting low-dimensional spaces

`dimensionality_reduction.py`: performs dimensionality reduction over PartDatasets using both MDS and t-SNE, and visualizes the resulting low-dimensional space.

`analyze_segment_resolutions.py`: Answers the question, "What % of segments are at `n` ticks per beat?"

`load_segment.py`: Load a single MIDI segment from a dataset.

`get_random_segments.py`: Select segments from a dataset at random. Just for exploring.

`plot_part_pair_dists.py`: Create various part pair distribution plots.

### Utility

`utils.py`: Generic helper functions used in multiple scripts

`notebook_utils.py`: Helper functions for notebooks
