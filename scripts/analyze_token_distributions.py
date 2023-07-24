import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader

from rhythmic_relationships import DATASETS_DIR, PLOTS_DIRNAME
from rhythmic_relationships.data import PartDataset


plots_dir = os.path.join(DATASETS_DIR, PLOTS_DIRNAME)
if not os.path.isdir(plots_dir):
    os.makedirs(plots_dir)

dataset_name = "lmdc_100_2bar_4res"

dataset_2 = PartDataset(
    dataset_name=dataset_name,
    part="Melody",
    representation="hits",
    block_size=2,
)
loader_2 = DataLoader(dataset_2, batch_size=len(dataset_2))
print(f"{len(dataset_2)=}")
hits_2 = next(iter(loader_2))


dataset_4 = PartDataset(
    dataset_name=dataset_name,
    part="Melody",
    representation="hits",
    block_size=2,
)
loader_4 = DataLoader(dataset_4, batch_size=len(dataset_4))
print(f"{len(dataset_4)=}")
hits_4 = next(iter(loader_4))
