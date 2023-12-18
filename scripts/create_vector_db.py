"""
Create a postgres vector db table for a dataset using embeddings from a trained model.
"""
import os

import psycopg
import torch
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from rhythmic_relationships import DATASETS_DIR, MODELS_DIR
from rhythmic_relationships.data import PartPairDataset
from rhythmic_relationships.model_utils import load_model
from rhythmic_relationships.models.hits_encdec import TransformerEncoderDecoder
from torch.utils.data import DataLoader

load_dotenv()

dataset_name = "lmdc_100_2bar_4res"
output_dir = os.path.join(DATASETS_DIR, dataset_name, "mashups")

# Load model
model_type = "hits_encdec"
model_name = "fragmental_2306210056"  # Melody -> Bass
model_dir = os.path.join(MODELS_DIR, model_type, model_name)

model_path = os.path.join(model_dir, "model.pt")

model, config = load_model(model_path, TransformerEncoderDecoder)
model = model.to("mps")

n_dim = config["model"]["context_len"] * config["model"]["enc_n_embed"]

# Load dataset
dataset = PartPairDataset(
    **{
        "dataset_name": dataset_name,
        "part_1": config["data"]["part_1"],
        "part_2": config["data"]["part_2"],
        "repr_1": "onset_roll",
        "repr_2": "onset_roll",
    }
)

loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
p1_seqs, _ = next(iter(loader))

segment_ids = dataset.loaded_segments.segment_id.values.tolist()

mtb_embeddings = model.encoder(p1_seqs.to("mps"), return_embeddings=True)

# DB connection parameters
connection_string = f"host=localhost dbname=pato user=postgres password={os.getenv('PGPASSWORD')} port=5432"

with psycopg.connect(connection_string) as conn:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)

    # Melody to Bass (mtb)
    conn.execute("DROP TABLE IF EXISTS segments_mtb")
    conn.execute(
        f"CREATE TABLE segments_mtb (id bigserial PRIMARY KEY,embedding vector({n_dim}))"
    )

    for mtb_embedding in mtb_embeddings:
        emb = (
            mtb_embedding.detach()
            .cpu()
            .view(
                n_dim,
            )
            .numpy()
        )
        conn.execute(
            "INSERT INTO segments_mtb (embedding) VALUES (%s)",
            (emb,),
        )

    # Check that the embeddings were inserted correctly
    neighbors = conn.execute(
        "SELECT * FROM segments_mtb ORDER BY embedding <-> %s LIMIT 3", (emb,)
    ).fetchall()

for neighbor in neighbors:
    print(neighbor[0])
