import logging
import os

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

DATASETS_DIR = os.path.join(THIS_DIR, "../output")

ANNOTATIONS_FILENAME = "segments.csv"
REPRESENTATIONS_DIRNAME = "representations"
PAIR_LOOKUPS_DIRNAME = "pair_lookups"
PLOTS_DIRNAME = "plots"
MODELS_DIR = os.path.join(DATASETS_DIR, "models")
INFERENCE_DIR = os.path.join(DATASETS_DIR, "inference")
CHECKPOINTS_DIRNAME = "checkpoints"
REPRESENTATIONS_FILENAME = "representations.txt"

WANDB_PROJECT_NAME = "rhythmic-relationships"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rhythmic-relationships")
