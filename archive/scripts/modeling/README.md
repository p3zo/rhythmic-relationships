
### Models

`train_part_pair_vae.py`: Train a conditional VAE over a PartPairDataset. Update `model_config.yml` to specify a new training configuration. Inference can be done in a general way with `inference_part_pair_vae.py` or with specific representations with `inference_part_pair_vae_pattern_to_pattern.py` and `inference_hits_to_hits.py`.

`train_part_vae.py`: Train a VAE for a PartDataset.

`bigrams.ipynb`: A notebook that walks through a bigram model for hits and chromas, following [makemore](https://github.com/karpathy/makemore).

`MLP.ipynb`: A notebook that walks through a finished trigram MLP model for chromas, following [makemore](https://github.com/karpathy/makemore).

`MLP_initial.ipynb`: A notebook that walks through initial explorations towards a trigram MLP model for chromas, following [makemore](https://github.com/karpathy/makemore).

`wavenet.ipynb`: A notebook that walks through a finished wavenet model for chromas, following [makemore](https://github.com/karpathy/makemore).

`model_utils.py`: Helper functions for modeling scripts

`predict.py`: Generate a bass roll given a drum roll.
