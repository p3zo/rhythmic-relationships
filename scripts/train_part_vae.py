import torch
import yaml
from model_utils import load_config, save_model
from rhythmic_relationships.data import PartDataset
from rhythmic_relationships.model import VariationalAutoEncoder
from rhythmic_relationships.train import train
from torch.utils.data import DataLoader

DEVICE = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
CHECKPOINTS_DIR = "models/checkpoints"
CONFIG_FILEPATH = "part_vae_config.yml"


if __name__ == "__main__":
    config = load_config(CONFIG_FILEPATH)
    print(yaml.dump(config))

    data = PartDataset(**config["dataset"])
    loader = DataLoader(data, batch_size=config["batch_size"], shuffle=True)

    model = VariationalAutoEncoder(**config["model"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    reduction = config["loss_reduction"]
    if config["loss_fn"] == "bce":
        loss_fn = torch.nn.BCELoss(reduction=reduction)
    elif config["loss_fn"] == "cross-entropy":
        loss_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    elif config["loss_fn"] == "mse":
        loss_fn = torch.nn.MSELoss(reduction=reduction)
    else:
        raise ValueError(f"`{config['loss_fn']}` is not a valid loss function")

    train(
        model=model,
        loader=loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
        device=DEVICE,
        checkpoints_dir=CHECKPOINTS_DIR,
    )

    samples = torch.randn(64, config["model"]["z_dim"]).to(DEVICE)
    sample = model.decode(samples).detach().cpu().numpy()
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from sklearn.manifold import TSNE

    print("Performing t-SNE...")
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=42)
    X_transform = tsne.fit_transform(sample)
    tsne_emb_df = pd.DataFrame(X_transform, columns=["component_1", "component_2"])
    sns.relplot(
        data=tsne_emb_df,
        x="component_1",
        y="component_2",
        height=8,
        aspect=1.25,
        legend=False,
    )
    plt.title(
        f"t-SNE of 64 samples from {config['dataset']['part']} latent space\n{config['dataset']['dataset_name']} {config['dataset']['representation']}"
    )
    plt.tight_layout()
    plt.savefig("sample_tsne.png")

    save_model(model, config)
