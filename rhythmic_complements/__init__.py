import logging
import os

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

DATASETS_DIR = os.path.join(THIS_DIR, "../output")
ANNOTATIONS_FILENAME = "segments.csv"
REPRESENTATIONS_DIRNAME = "representations"
PAIR_LOOKUPS_DIRNAME = "pair_lookups"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rhythmic-complements")
