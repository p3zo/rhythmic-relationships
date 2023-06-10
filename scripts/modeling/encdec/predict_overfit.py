import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from rhythmic_relationships.model_utils import load_model
from rhythmic_relationships import MODELS_DIR, CHECKPOINTS_DIRNAME
from rhythmic_relationships.data import get_roll_from_sequence, tokenize_roll
from rhythmic_relationships.io import write_midi_from_roll, load_midi_file, slice_midi
from rhythmic_relationships.models.encdec import TransformerEncoderDecoder
from rhythmic_relationships.vocab import get_vocab_encoder_decoder
from rhythmtoolbox import pianoroll2descriptors


MODEL_NAME = "unpropriety_2305301224"
CHECKPOINT_NUM = None

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_built()
    else torch.device("cuda:0")
    if torch.cuda.device_count() > 0
    else torch.device("cpu")
)


def load_roll_from_file(fp, part):
    # TODO: simplify
    pmid = load_midi_file(fp, resolution=24)
    sp_reprs = slice_midi(
        pmid,
        seg_size=2,
        resolution=24,
        n_beat_bars=4,
        min_seg_pitches=1,
        min_seg_beats=1,
    )
    if not len(sp_reprs) == 1:
        return None
    if part == "Drums":
        raise NotImplementedError
    onset_roll_repr_ix = 1
    return sp_reprs[list(sp_reprs.keys())[0]][0][onset_roll_repr_ix]


def load_src_tgt_rolls(model_dir, src_part, tgt_part):
    # TODO: simplify
    mids = glob.glob(os.path.join(model_dir, "*.mid"))
    src_rolls = []
    tgt_rolls = []
    for fp in sorted(mids):
        if "_tgt" in fp:
            roll = load_roll_from_file(fp, tgt_part)
            if roll is not None:
                tgt_rolls.append(roll.tolist())
        elif "_src" in fp:
            roll = load_roll_from_file(fp, src_part)
            if roll is not None:
                src_rolls.append(roll.tolist())
        else:
            raise ValueError('File name must contain either "_src" or "_tgt"')

    return torch.LongTensor(src_rolls), torch.LongTensor(tgt_rolls)


if __name__ == "__main__":
    model_dir = os.path.join(MODELS_DIR, MODEL_NAME)

    if CHECKPOINT_NUM:
        checkpoints_dir = os.path.join(model_dir, CHECKPOINTS_DIRNAME)
        model_path = os.path.join(checkpoints_dir, str(CHECKPOINT_NUM))
    else:
        model_path = os.path.join(model_dir, "model.pt")

    model, config = load_model(model_path, TransformerEncoderDecoder)
    model = model.to(DEVICE)

    part_1 = config["data"]["part_1"]
    part_2 = config["data"]["part_2"]
    n_ticks = config["sequence_len"]

    gen_dir = os.path.join(MODELS_DIR, MODEL_NAME, "inference")
    if not os.path.isdir(gen_dir):
        os.makedirs(gen_dir)

    # Generate seqs using part_1s from the dataset as encoder input and a start token as decoder input
    desc_dfs = []
    n_generated = 0
    all_zeros = 0

    src_rolls, tgt_rolls = load_src_tgt_rolls(
        model_dir,
        src_part=config["data"]["part_1"],
        tgt_part=config["data"]["part_2"],
    )

    n_seqs = len(src_rolls)
    encode, _ = get_vocab_encoder_decoder(config["data"]["part_2"])
    start_ix = encode(["start"])[0]
    idy = torch.full((n_seqs, 1), start_ix, dtype=torch.long, device=DEVICE)

    srcs = torch.LongTensor(
        [tokenize_roll(i.numpy(), config["data"]["part_1"]) for i in src_rolls]
    ).to(DEVICE)
    tgts = torch.LongTensor(
        [tokenize_roll(i.numpy(), config["data"]["part_2"]) for i in tgt_rolls]
    ).to(DEVICE)

    seqs = (
        model.generate(srcs, idy, max_new_tokens=config["sequence_len"]).cpu().numpy()
    )

    srcs = srcs.cpu().numpy()
    tgts = tgts.cpu().numpy()

    for ix, seq in enumerate(seqs):
        gen_roll = get_roll_from_sequence(seq, part=config["data"]["part_2"])

        src_roll = src_rolls[ix].numpy()
        tgt_roll = tgt_rolls[ix].numpy()

        write_midi_from_roll(
            src_roll,
            outpath=os.path.join(gen_dir, f"{ix}_src.mid"),
            part=part_1,
            binary=False,
            onset_roll=True,
        )
        write_midi_from_roll(
            tgt_roll,
            outpath=os.path.join(gen_dir, f"{ix}_tgt.mid"),
            part=part_2,
            binary=False,
            onset_roll=True,
        )

        # Compare descriptors of the generated and target rolls
        gen_roll_descs = pianoroll2descriptors(
            gen_roll,
            config["resolution"],
            drums=part_2 == "Drums",
        )
        tgt_roll_descs = pianoroll2descriptors(
            tgt_roll,
            config["resolution"],
            drums=part_2 == "Drums",
        )
        df = pd.DataFrame.from_dict(
            {"generated": gen_roll_descs, "target": tgt_roll_descs}, orient="index"
        )
        desc_dfs.append(df)

        n_generated += 1
        if gen_roll.max() == 0:
            all_zeros += 1
            continue

        write_midi_from_roll(
            gen_roll,
            outpath=os.path.join(gen_dir, f"{ix}_gen.mid"),
            part=part_2,
            binary=False,
            onset_roll=True,
        )

    print(f"{n_generated=}, {all_zeros=} ({100*round(all_zeros/n_generated, 2)}%)")
