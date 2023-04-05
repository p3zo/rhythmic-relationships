## Scripts

### Data

`prepare_data.py`: Create a new dataset. See instructions in [Create the dataset](../README.md#create-the-dataset)

`create_chroma_vocabulary.ipynb`: A notebook that creates a reduced vocabulary of chord types to use for modeling
chromas.

### Models

`train_part_pair_vae.py`: Train a conditional VAE over a PartPairDataset. Update `model_config.yml` to specify a new
training configuration. Inference can be done in a general way with `inference_part_pair_vae.py` or with specific
representations with `inference_part_pair_vae_pattern_to_pattern.py` and `inference_hits_to_hits.py`.

`train_part_vae.py`: Train a VAE for a PartDataset.

`bigrams.ipynb`: A notebook that walks through a bigram model for hits and chromas,
following [makemore](https://github.com/karpathy/makemore).

`MLP.ipynb`: A notebook that walks through a finished trigram MLP model for chromas,
following [makemore](https://github.com/karpathy/makemore).

`MLP_initial.ipynb`: A notebook that walks through initial explorations towards a trigram MLP model for chromas,
following [makemore](https://github.com/karpathy/makemore).

`wavenet.ipynb`: A notebook that walks through a finished wavenet model for chromas,
following [makemore](https://github.com/karpathy/makemore).

### Analysis

`pairspace.py`: performs dimensionality reduction over PartPairDatasets using both MDS and t-SNE, and
visualizes the resulting low-dimensional spaces

`dimensionality_reduction.py`: performs dimensionality reduction over PartDatasets using both MDS and t-SNE, and
visualizes the resulting low-dimensional space.

`analyze_segment_resolutions.py`: Answers the question, "What % of segments are at `n` ticks per beat?"

`get_random_segments.py`: Select segments from a dataset at random. Just for exploring.

`plot_part_pair_dists.py`: Create various part pair distribution plots.

### Utility

`utils.py`: Generic helper functions used in multiple scripts

`model_utils.py`: Helper functions for modeling scripts

`notebook_utils.py`: Helper functions for notebooks

#### DEFUNCT

These scripts are from outdated experiments and are likely no longer functional.

`predict.py`: Generate a bass roll given a drum roll.

`analyze_segroll_samples.py`: Compare the descriptor distributions of many latent samples to the distributions of the
input data
