"""Export trained hits encoder-decoders into a static site's worth of files.

GitHub Pages serves files, not Python, so everything the interface needs at runtime has to be
baked out ahead of time: each model as ONNX for onnxruntime-web, a pool of real input segments,
and the segment index the nearest-rhythm search reads. The dataset itself stays where it is.

Segments are keyed by part rather than by model, because a part means the same thing to every
model that reads it: Melody->Bass and Harmony->Bass search one Bass index between them. The page
loads only what the selected model needs.

The int8 export is the one that ships. It is checked here against the original weights by how
often it would make the same greedy choice, because that - not the size of a logit difference -
is what decides whether the site generates the same rhythms the model does.
"""

import argparse
import glob
import json
import os
import random
import shutil

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from rhythmic_relationships import (
    DATASETS_DIR,
    REPRESENTATIONS_DIRNAME,
    REPRESENTATIONS_FILENAME,
)
from rhythmic_relationships.data import load_co_occurring_hits
from rhythmic_relationships.interlock import (
    fit_pair_score,
    fit_relationship_targets,
    lineup_accuracy,
)
from rhythmic_relationships.model_utils import load_model
from rhythmic_relationships.models.hits_decoder import get_causal_mask
from rhythmic_relationships.models.hits_encdec import TransformerEncoderDecoder
from rhythmic_relationships.vocab import START_IX, get_hits_vocab

import onnxruntime as ort


class EncoderOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, src):
        return self.encoder(src, return_embeddings=True)


class DecoderOnly(torch.nn.Module):
    """The decoder over a full-length target.

    Generation in the browser refills a fixed 32-step buffer rather than growing a sequence,
    which is equivalent under a causal mask: step t only ever attends to steps 0..t, so whatever
    padding sits after it cannot reach it. Keeping the length fixed means no dynamic shapes.
    """

    def __init__(self, model, n_steps):
        super().__init__()
        self.decoder = model.decoder
        self.n_steps = n_steps

    def forward(self, tgt, enc):
        mask = get_causal_mask(self.n_steps, device=tgt.device, boolean=True)
        return self.decoder(tgt, attn_mask=mask, context=enc)


def export_onnx(model, n_steps, outdir, quantize):
    src = torch.randint(2, 7, (1, n_steps))
    batch_axis = {0: "batch"}

    encoder_path = os.path.join(outdir, "encoder.onnx")
    torch.onnx.export(
        EncoderOnly(model), (src,), encoder_path,
        input_names=["src"], output_names=["enc"],
        dynamic_axes={"src": batch_axis, "enc": batch_axis},
        opset_version=17, dynamo=False,
    )
    # The exporter restores the wrapper's original training flag onto its children, which would
    # leave dropout live for everything that follows
    model.eval()

    with torch.no_grad():
        enc = EncoderOnly(model)(src)
    decoder_path = os.path.join(outdir, "decoder.onnx")
    torch.onnx.export(
        DecoderOnly(model, n_steps), (torch.randint(2, 7, (1, n_steps)), enc), decoder_path,
        input_names=["tgt", "enc"], output_names=["logits"],
        dynamic_axes={"tgt": batch_axis, "enc": batch_axis, "logits": batch_axis},
        opset_version=17, dynamo=False,
    )
    model.eval()

    paths = {"encoder": encoder_path, "decoder": decoder_path}
    if quantize:
        for name, path in list(paths.items()):
            quantized = path.replace(".onnx", ".int8.onnx")
            quantize_dynamic(path, quantized, weight_type=QuantType.QUInt8)
            print(f"  {name}: {os.path.getsize(path)/1e6:.1f} MB -> "
                  f"{os.path.getsize(quantized)/1e6:.1f} MB int8")
            os.remove(path)
            shutil.move(quantized, path)
    return paths


def check_fidelity(model, paths, n_steps, n_seqs=64):
    """How often the exported model would make the same greedy choice as the original."""
    torch.manual_seed(0)
    src = torch.randint(2, 7, (n_seqs, n_steps))
    tgt = torch.randint(2, 7, (n_seqs, n_steps))
    with torch.no_grad():
        reference = model(src=src, tgt=tgt).numpy()

    encoder = ort.InferenceSession(paths["encoder"])
    decoder = ort.InferenceSession(paths["decoder"])
    enc = encoder.run(None, {"src": src.numpy()})[0]
    logits = decoder.run(None, {"tgt": tgt.numpy(), "enc": enc})[0]

    agree = float((logits.argmax(-1) == reference.argmax(-1)).mean())
    print(f"  exported vs original: greedy choice matches on {agree * 100:.1f}% of "
          f"{n_seqs * n_steps} steps, max |delta logit| {np.abs(logits - reference).max():.3f}")
    return agree


def collect_segments(dataset_name, part, n_wanted, seed, per_file, with_pitches=True):
    dataset_dir = os.path.join(DATASETS_DIR, dataset_name)
    with open(os.path.join(dataset_dir, REPRESENTATIONS_FILENAME)) as f:
        names = f.read().split(",")
    hits_ix, onset_roll_ix = names.index("hits"), names.index("onset_roll")

    paths = glob.glob(
        os.path.join(dataset_dir, REPRESENTATIONS_DIRNAME, "**", "*.npz"), recursive=True
    )
    rng = random.Random(seed)
    rng.shuffle(paths)

    out = []
    for path in paths:
        with np.load(path, allow_pickle=True) as npz:
            # A three-minute song holds dozens of two-bar segments. Taking them in order fills
            # the whole quota from a handful of songs, so the page offers the same few pieces of
            # music over and over; take a sample spread across the file instead.
            keys = [k for k in npz.files if k.endswith(f"_{part}")]
            for key in rng.sample(keys, min(per_file, len(keys))):
                reprs = npz[key][0]
                # Velocities are quantised to the model's four bins, so a byte each is lossless
                hits = [int(round(float(v) * 4)) for v in reprs[hits_ix]]
                entry = {"h": hits, "n": f"{os.path.basename(path)[:-4]} seg {key.split('_')[0]}"}
                if with_pitches:
                    onset_roll = np.asarray(reprs[onset_roll_ix])
                    steps, pitches = np.nonzero(onset_roll)
                    entry["p"] = [
                        [int(s), int(p), int(round(float(onset_roll[s, p]) * 127))]
                        for s, p in zip(steps, pitches)
                    ]
                out.append(entry)
                if len(out) >= n_wanted:
                    return out
    return out


def fit_relationships(dataset_name, part_1, part_2, n_pairs, n_targets, n_imposters, seed):
    """Relationships real pairs of these parts had, for the page to retrieve toward.

    The page needs these because the generated rhythm turned out not to be a useful query: the
    ablation found a mashup picked to match it no closer to a real relationship than one picked at
    random. What replaces it is not a score to maximise - that picks whatever doubles the input -
    but a target drawn from real pairs, with candidates ranked by how close they come to it.

    The lineup accuracy printed here is not what the page claims; it is the evidence that these
    features carry relationship information at all, without which aiming at a target in their space
    would mean nothing. It is measured on half the pairs after fitting a discriminant on the other
    half, while the targets that ship come from all of them.
    """
    pairs = load_co_occurring_hits(dataset_name, [part_1, part_2], n_pairs, seed, per_file=1)
    a, b = pairs[part_1], pairs[part_2]
    half = len(a) // 2
    held_out = fit_pair_score(a[:half], b[:half], seed)
    accuracy, n = lineup_accuracy(held_out, a[half:], b[half:],
                                  np.random.default_rng(seed + 1), n_imposters)
    print(f"  interlock features: tell a real partner from an imposter on {accuracy * 100:.1f}% "
          f"of {n} lineups they were not fitted on")

    targets, precision, means = fit_relationship_targets(a, b, n_targets, seed)
    print(f"  {len(targets)} target relationships; real pairs land on "
          f"{means[0] * 100:.0f}% of each other's onsets")
    return {
        "targets": np.round(targets, 5).tolist(),
        "precision": np.round(precision, 5).tolist(),
        "real_means": np.round(means, 5).tolist(),
        "lineup_accuracy": round(accuracy, 4),
        "n_pairs": len(a),
    }


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  {os.path.relpath(path)}  {os.path.getsize(path)/1e6:.2f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, nargs="+")
    parser.add_argument("--outdir", type=str, default="docs")
    parser.add_argument("--examples", type=int, default=2000)
    parser.add_argument("--index", type=int, default=30000)
    # One example per song, so "Random real input" keeps offering something new. The index can
    # afford a few per song: the search already caps its results at one per song.
    parser.add_argument("--examples_per_file", type=int, default=1)
    parser.add_argument("--index_per_file", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--fit_pairs", type=int, default=2000,
                        help="Real pairs to read relationships off. A few hundred is too few: a "
                             "discriminant fitted on 100 pairs scores at chance")
    parser.add_argument("--targets", type=int, default=256,
                        help="Target relationships shipped per pair, for the page to draw from")
    parser.add_argument("--fit_imposters", type=int, default=40)
    parser.add_argument("--no_quantize", action="store_true")
    args = parser.parse_args()

    data_dir = os.path.join(args.outdir, "data")
    os.makedirs(data_dir, exist_ok=True)

    models, shared, parts_wanted = [], {}, {}
    for model_path in args.model_path:
        model, config = load_model(model_path, TransformerEncoderDecoder)
        model.eval()
        run = os.path.basename(os.path.dirname(model_path))
        part_1, part_2 = config["data"]["part_1"], config["data"]["part_2"]
        n_steps = config["model"]["context_len"]

        # One page holds one sequencer and one tokenizer, so every model on it has to agree
        # about the representation. Anything else belongs on its own page.
        facts = {"n_steps": n_steps, "dataset": config["data"]["dataset_name"],
                 "block_size": config["data"]["block_size"]}
        if shared and facts != shared:
            raise ValueError(f"{run} does not match the others: {facts} vs {shared}")
        shared = facts

        print(f"exporting {run}  ({part_1} -> {part_2})")
        model_dir = os.path.join(data_dir, "models", run)
        os.makedirs(model_dir, exist_ok=True)
        paths = export_onnx(model, n_steps, model_dir, quantize=not args.no_quantize)
        agree = check_fidelity(model, paths, n_steps)

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        evals = checkpoint.get("evals") or []
        relationships = fit_relationships(config["data"]["dataset_name"], part_1, part_2,
                                          args.fit_pairs, args.targets, args.fit_imposters,
                                          args.seed)
        models.append({
            "id": run,
            "part_1": part_1,
            "part_2": part_2,
            "relationships": relationships,
            "n_params": sum(p.numel() for p in model.parameters()),
            "val_loss": round(evals[-1]["val_loss"], 4) if evals else None,
            "epochs": checkpoint.get("epoch"),
            "greedy_agreement": round(agree, 4),
        })
        parts_wanted.setdefault(part_1, set()).add("examples")
        parts_wanted.setdefault(part_2, set()).add("index")

    dataset_name = shared["dataset"]
    parts = {}
    for part, wanted in sorted(parts_wanted.items()):
        part_dir = os.path.join(data_dir, "parts", part)
        os.makedirs(part_dir, exist_ok=True)
        parts[part] = {}
        if "examples" in wanted:
            print(f"collecting {args.examples} {part} inputs")
            segs = collect_segments(dataset_name, part, args.examples, args.seed,
                                    args.examples_per_file)
            write_json(os.path.join(part_dir, "examples.json"), segs)
            parts[part]["n_examples"] = len(segs)
        if "index" in wanted:
            print(f"collecting {args.index} {part} segments to search")
            segs = collect_segments(dataset_name, part, args.index, args.seed + 1,
                                    args.index_per_file)
            write_json(os.path.join(part_dir, "index.json"), segs)
            parts[part]["n_index"] = len(segs)

    hits_vocab = get_hits_vocab()
    meta = {
        "dataset": dataset_name,
        "n_steps": shared["n_steps"],
        "resolution": 4,
        "n_beat_bars": 4,
        # The pitch a drawn rhythm is sounded at, for any part. It carries no musical claim -
        # a hand-drawn grid has no pitch content of its own.
        "input_pitch": 55,
        "start_ix": START_IX,
        # token id -> hit value, so the page can encode and decode exactly as training did
        "hits_tokens": {str(k): v for k, v in hits_vocab.items() if not isinstance(v, str)},
        # The interlock features the fitted weights above multiply, in order. The page ports
        # these and is checked against the Python by scripts/check_web_port.py.
        "interlock_features": [
            "of the candidate's onsets, the share landing with the input",
            "of the input's onsets, the share the candidate leaves alone",
            "how much of the bar neither part touches",
        ],
        "models": sorted(models, key=lambda m: (m["part_1"], m["part_2"])),
        "parts": parts,
    }
    write_json(os.path.join(data_dir, "meta.json"), meta)

    total = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(data_dir)
        for f in files
        if not f.startswith(".")
    )
    print(f"\n{data_dir}: {total/1e6:.1f} MB total")


if __name__ == "__main__":
    main()
