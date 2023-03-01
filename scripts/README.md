## Scripts

`prepare_data.py`: Create a new dataset. See instructions in [Create the dataset](../README.md#create-the-dataset)

`train_0a.py`: Train a model. Update `model_config.yml` to specify a new training configuration.

`inference_pattern_to_pattern.py`: Load a pattern_to_pattern model and perform inference using a specified input.
Use `model_name` in the script to point to the model to use.

`inference_hits_to_hits.py`: Load a hits_to_hits model and perform inference using a specified input. Use `model_name`
in the script to point to the model to use.

`inference_0a.py`: Unfinished, generalized version of an inference script. Load any model and perform inference using a
randomly sampled observation from the dataset. Use `model_name` in the script to point to the model to use.

`analyze_segment_resolutions.py`: Answers the question, "What % of segments are at `n` ticks per beat?"

#### DEFUNCT

These scripts are from outdated experiments and are likely no longer functional.

`predict.py`: Generate a bass roll given a drum roll.

`rsp_pairs_train.py`: Train a conditional VAE on pairs of rhythm space points (RSPs).

`rsp_train.py`: Train a VAE for a single part using a rhythm descriptor representation. Example inference is given
in `rsp_inference.py`.

`segroll_train.py`: Train a VAE for a single part using a piano roll representation. Example inference is given
in `segroll_inference.py`.

`analyze_segroll_samples.py`: Compare the descriptor distributions of many latent samples to the distributions of the
input data
