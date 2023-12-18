"""
Mashup segments using rhythmic relationships.
Stitch the melody from one segment with the bassline of a second.
"""
import os
import music21
import copy
import pandas as pd
import torch
from rhythmic_relationships import DATASETS_DIR
from rhythmic_relationships.data import get_roll_from_sequence, PartPairDataset
from rhythmic_relationships.io import write_midi_from_roll_list
from torch.utils.data import DataLoader

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from rhythmic_relationships import MODELS_DIR
from rhythmic_relationships.model_utils import load_model
from rhythmic_relationships.models.hits_encdec import TransformerEncoderDecoder
from rhythmic_relationships.evaluate import hits_inference


dataset_name = "lmdc_100_2bar_4res"
part_1 = "Melody"
part_2 = "Bass"
n_mixes = 10
output_dir = os.path.join(DATASETS_DIR, dataset_name, "mashups", f"{part_1}_{part_2}")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def roll_to_music21_stream(piano_roll, time_step=0.25):
    """
    Convert a piano roll to a music21 stream.

    :param piano_roll: A 2D NumPy array representing the piano roll.
                       Rows correspond to pitches (MIDI note numbers), and columns to time steps.
    :param time_step: Duration of each time step in the piano roll, in quarter notes.
    :return: A music21 stream object representing the piano roll.
    """
    s = music21.stream.Stream()

    # Track the start time of the current note for each pitch
    note_start_times = {pitch: None for pitch in range(piano_roll.shape[0])}

    # Iterate over time steps in the piano roll
    for col in range(piano_roll.shape[1] + 1):  # +1 to handle the ending of notes
        for pitch in range(piano_roll.shape[0]):
            is_note_on = col < piano_roll.shape[1] and piano_roll[pitch, col]

            if is_note_on and note_start_times[pitch] is None:
                # Note start
                note_start_times[pitch] = col * time_step
            elif not is_note_on and note_start_times[pitch] is not None:
                # Note end
                note_duration = col * time_step - note_start_times[pitch]
                n = music21.note.Note()
                n.pitch.midi = pitch
                n.duration.quarterLength = note_duration
                n.offset = note_start_times[pitch]
                s.insert(n.offset, n)
                note_start_times[pitch] = None

    return s


def transpose_stream(stream, from_key, to_key):
    """stream is a music21 stream
    TODO: unit test"""

    # Convert pretty-midi key numbers to music21 keys
    key_names = [
        "C Major",
        "Db Major",
        "D Major",
        "Eb Major",
        "E Major",
        "F Major",
        "Gb Major",
        "G Major",
        "Ab Major",
        "A Major",
        "Bb Major",
        "B Major",
        "C minor",
        "C# minor",
        "D minor",
        "Eb minor",
        "E minor",
        "F minor",
        "F# minor",
        "G minor",
        "G# minor",
        "A minor",
        "Bb minor",
        "B minor",
    ]

    original_key = music21.key.Key(*key_names[from_key].split(" "))
    target_key = music21.key.Key(*key_names[to_key].split(" "))

    interval_to_transpose = music21.interval.Interval(
        original_key.tonic, target_key.tonic
    )

    return stream.transpose(interval_to_transpose)


def merge_m21_streams(stream1, stream2):
    """
    Merge two music21 streams into separate parts within a single score,
    preserving offsets and durations.

    :param stream1: The first music21 stream.
    :param stream2: The second music21 stream.
    :return: A music21 score with two parts.
    """
    # Create a new score
    score = music21.stream.Score()

    # Create two new parts
    part1 = music21.stream.Part()
    part2 = music21.stream.Part()

    # Copy elements from stream1 to part1, preserving offsets
    for element in stream1:
        cloned_element = copy.deepcopy(element)
        part1.insert(element.offset, cloned_element)

    # Copy elements from stream2 to part2, preserving offsets
    for element in stream2:
        cloned_element = copy.deepcopy(element)
        part2.insert(element.offset, cloned_element)

    # Add the parts to the score
    score.insert(0, part1)
    score.insert(0, part2)

    return score


dataset = PartPairDataset(
    **{
        "dataset_name": dataset_name,
        "part_1": part_1,
        "part_2": part_2,
        "repr_1": "onset_roll",
        "repr_2": "onset_roll",
    }
)

loader = DataLoader(dataset, batch_size=n_mixes, shuffle=True)

p1_seqs, p2_seqs = next(iter(loader))
dataset.loaded_segments.to_csv(os.path.join(output_dir, "loaded_segments.csv"))

key_list = dataset.loaded_segments.key.values

# Write originals
for ix, (i, j) in enumerate(zip(p1_seqs, p2_seqs)):
    i_roll = get_roll_from_sequence(i.numpy(), part_1)
    j_roll = get_roll_from_sequence(j.numpy(), part_2)

    outpath = os.path.join(output_dir, f"{ix}.mid")

    write_midi_from_roll_list(
        [i_roll, j_roll],
        outpath,
        binary=False,
        onset_roll=True,
        parts=[part_1, part_2],
    )


# Mashup method 1: Random
ixs = torch.randperm(n_mixes).tolist()
p2_seqs_mixed = p2_seqs[ixs]

random_output_dir = os.path.join(output_dir, "random")
if not os.path.exists(random_output_dir):
    os.mkdir(random_output_dir)

random_mashup_meta = pd.DataFrame()

for ix, (i, j) in enumerate(zip(p1_seqs, p2_seqs_mixed)):
    i_roll = get_roll_from_sequence(i.numpy(), part_1)
    j_roll = get_roll_from_sequence(j.numpy(), part_2)

    i_key = key_list[ix]
    j_key = key_list[ixs[ix]]

    p1_segment = dataset.loaded_segments.iloc[ix]
    p2_segment = dataset.loaded_segments.iloc[ixs[ix]]

    p1_segment["part"] = part_1
    p2_segment["part"] = part_2

    merged_meta = (
        p1_segment.to_frame()
        .T.reset_index()
        .join(p2_segment.to_frame().T.reset_index(), lsuffix="_p1", rsuffix="_p2")
    )
    random_mashup_meta = pd.concat([random_mashup_meta, merged_meta], axis=0)

    j_stream = roll_to_music21_stream(j_roll.T)
    j_stream_transposed = transpose_stream(j_stream, j_key, i_key)

    i_stream = roll_to_music21_stream(i_roll.T)
    merged_streams = merge_m21_streams(i_stream, j_stream_transposed)

    outpath = os.path.join(random_output_dir, f"{ix}.mid")

    merged_streams.write("midi", outpath)

    # write_midi_from_roll_list(
    #     [i_roll, j_roll],
    #     outpath,
    #     binary=False,
    #     onset_roll=True,
    #     parts=[part_1, part_2],
    # )

random_mashup_meta.to_csv(os.path.join(random_output_dir, "meta.csv"), index=False)


# Mashup method 2: Using rhythm Model

load_dotenv()

dataset_name = "lmdc_100_2bar_4res"
output_dir = os.path.join(DATASETS_DIR, dataset_name, "mashups")

# Load Melody -> Bass model
model_path = os.path.join(
    MODELS_DIR, "hits_encdec", "fragmental_2306210056", "model.pt"
)
model, config = load_model(model_path, TransformerEncoderDecoder)
model = model.to("mps")

n_dim = config["model"]["context_len"] * config["model"]["enc_n_embed"]


mtb_embeddings = []
for src in p1_seqs.to("mps"):
    seq = hits_inference(
        model=model,
        src=src.unsqueeze(0),
        n_tokens=config["model"]["context_len"],
        temperature=1,
        device="mps",
        sampler="nucleus",
        nucleus_p=0.92,
    )
    mtb_embeddings.append(
        model.encoder(seq.unsqueeze(0), return_embeddings=True)
        .detach()
        .cpu()
        .view(n_dim)
        .numpy()
    )

connection_string = f"host=localhost dbname=pato user=postgres password={os.getenv('PGPASSWORD')} port=5432"

emb_neighbors = []
with psycopg.connect(connection_string) as conn:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)

    for emb in mtb_embeddings:
        neighbors = conn.execute(
            "SELECT * FROM segments_mtb ORDER BY embedding <-> %s LIMIT 1",
            (emb,),
        ).fetchall()
        emb_neighbors.append(neighbors)

for ix, en in enumerate(emb_neighbors):
    for neighbor in en:
        print(ix, neighbor[0])
