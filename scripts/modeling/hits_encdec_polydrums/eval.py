"""Compare training set descriptors to generated after each epoch"""
import os

import torch
from rhythmic_relationships import CHECKPOINTS_DIRNAME, MODELS_DIR
from rhythmic_relationships.data import PartDataset, PartPairDataset
from rhythmic_relationships.evaluate import get_flat_nonzero_dissimilarity_matrix
from rhythmic_relationships.model_utils import load_model
from rhythmic_relationships.models.hits_encdec import TransformerEncoderDecoder
from torch.utils.data import DataLoader
from train import eval_gen_hits_encdec

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_built()
    else torch.device("cuda:0")
    if torch.cuda.device_count() > 0
    else torch.device("cpu")
)


if __name__ == "__main__":
    model_type = "hits_encdec_polydrums"
    model_name = "overclog_2306281254"
    checkpoint_num = None

    n_training_obs = 100
    n_eval_seqs = 25
    pitch = 72
    resolution = 4
    temperature = 1
    nucleus_p = 0.92
    samplers = ["multinomial", "nucleus"]

    model_dir = os.path.join(MODELS_DIR, model_type, model_name)

    eval_dir = os.path.join(model_dir, "eval")
    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)

    gen_dir = os.path.join(model_dir, "inference")
    if not os.path.isdir(gen_dir):
        os.makedirs(gen_dir)

    device = DEVICE

    if checkpoint_num:
        checkpoints_dir = os.path.join(model_dir, CHECKPOINTS_DIRNAME)
        model_path = os.path.join(checkpoints_dir, str(checkpoint_num))
    else:
        model_path = os.path.join(model_dir, "model.pt")

    model, config = load_model(model_path, TransformerEncoderDecoder)
    model = model.to(DEVICE)
    part_1 = config["data"]["part_1"]
    part_2 = config["data"]["part_2"]
    drop_cols = ["noi", "polyDensity", "syness"]

    # Get distribution from training set
    full_df = PartDataset(
        dataset_name=config["data"]["dataset_name"],
        part=part_2,
        representation="descriptors",
    ).as_df(subset=n_training_obs)
    dataset_df = full_df.drop(
        ["filename", "segment_id"] + drop_cols,
        axis=1,
    ).dropna(how="all", axis=1)

    train_dist = get_flat_nonzero_dissimilarity_matrix(dataset_df.values)

    # Load data for inference
    # TODO: load only data from val split
    dataset = PartPairDataset(**config["data"], tokenize_rolls=True)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    sampled = eval_gen_hits_encdec(
        model=model,
        config=config,
        loader=loader,
        n_seqs=n_eval_seqs,
        eval_dir=eval_dir,
        model_name=model_name,
        device=device,
        train_df=dataset_df,
        train_dist=train_dist,
        samplers=samplers,
    )
