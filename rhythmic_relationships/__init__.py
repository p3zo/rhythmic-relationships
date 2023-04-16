import logging
import os

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

DATASETS_DIR = os.path.join(THIS_DIR, "../output")

ANNOTATIONS_FILENAME = "segments.csv"
REPRESENTATIONS_DIRNAME = "representations"
PAIR_LOOKUPS_DIRNAME = "pair_lookups"
PLOTS_DIRNAME = "plots"
MODELS_DIRNAME = "models"
CHECKPOINTS_DIRNAME = "checkpoints"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rhythmic-relationships")
