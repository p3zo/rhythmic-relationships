## Scripts

`prepare_data.py`: Create a new dataset. See instructions in [Create the dataset](../README.md#create-the-dataset)

`dimensionality_reduction.py`: performs dimensionality reduction over PartDatasets using both MDS and t-SNE, and
visualizes the resulting low-dimensional space.

`train_0a.py`: Train a model. Update `model_config.yml` to specify a new training configuration.

`inference_pattern_to_pattern.py`: Load a pattern_to_pattern model and perform inference using a specified input.
Use `model_name` in the script to point to the model to use.

`inference_hits_to_hits.py`: Load a hits_to_hits model and perform inference using a specified input. Use `model_name`
in the script to point to the model to use.

`inference_0a.py`: Unfinished, generalized version of an inference script. Load any model and perform inference using a
randomly sampled observation from the dataset. Use `model_name` in the script to point to the model to use.

`analyze_segment_resolutions.py`: Answers the question, "What % of segments are at `n` ticks per beat?"

`plot_part_pair_dists.py`: Create various part pair distribution plots.

## Notebooks

`create_chroma_vocabulary.ipynb`: A notebook that creates a reduced vocabulary of chord types to use for modeling
chromas.

`bigrams.ipynb`: A notebook that walks through a bigram model for hits and chromas,
following [makemore](https://github.com/karpathy/makemore).

`MLP.ipynb`: A notebook that walks through a finished trigram MLP model for chromas,
following [makemore](https://github.com/karpathy/makemore).

`MLP_initial.ipynb`: A notebook that walks through initial explorations towards a trigram MLP model for chromas,
following [makemore](https://github.com/karpathy/makemore).

#### DEFUNCT

These scripts are from outdated experiments and are likely no longer functional.

`predict.py`: Generate a bass roll given a drum roll.

`analyze_segroll_samples.py`: Compare the descriptor distributions of many latent samples to the distributions of the
input data
