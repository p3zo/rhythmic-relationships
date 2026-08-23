"""Export a trained hits encoder-decoder into a static site's worth of files.

GitHub Pages serves files, not Python, so everything the interface needs at runtime has to be
baked out ahead of time: the model as ONNX for onnxruntime-web, a handful of real input segments,
and the segment index the nearest-rhythm search reads. The dataset itself stays where it is.

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


def collect_segments(dataset_name, part, n_wanted, seed, with_pitches=True):
    dataset_dir = os.path.join(DATASETS_DIR, dataset_name)
    with open(os.path.join(dataset_dir, REPRESENTATIONS_FILENAME)) as f:
        names = f.read().split(",")
    hits_ix, onset_roll_ix = names.index("hits"), names.index("onset_roll")

    paths = glob.glob(
        os.path.join(dataset_dir, REPRESENTATIONS_DIRNAME, "**", "*.npz"), recursive=True
    )
    random.Random(seed).shuffle(paths)

    out = []
    for path in paths:
        with np.load(path, allow_pickle=True) as npz:
            for key in [k for k in npz.files if k.endswith(f"_{part}")]:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="docs")
    parser.add_argument("--examples", type=int, default=200)
    parser.add_argument("--index", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no_quantize", action="store_true")
    args = parser.parse_args()

    model, config = load_model(args.model_path, TransformerEncoderDecoder)
    model.eval()
    n_steps = config["model"]["context_len"]
    part_1 = config["data"]["part_1"]
    part_2 = config["data"]["part_2"]
    dataset_name = config["data"]["dataset_name"]

    data_dir = os.path.join(args.outdir, "data")
    os.makedirs(data_dir, exist_ok=True)

    print("exporting onnx")
    paths = export_onnx(model, n_steps, data_dir, quantize=not args.no_quantize)
    agree = check_fidelity(model, paths, n_steps)

    print(f"collecting {args.examples} {part_1} inputs")
    examples = collect_segments(dataset_name, part_1, args.examples, args.seed)
    print(f"collecting {args.index} {part_2} segments to search")
    index = collect_segments(dataset_name, part_2, args.index, args.seed + 1)

    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    evals = checkpoint.get("evals") or []
    hits_vocab = get_hits_vocab()

    meta = {
        "run": os.path.basename(os.path.dirname(args.model_path)),
        "part_1": part_1,
        "part_2": part_2,
        "dataset": dataset_name,
        "n_steps": n_steps,
        "resolution": 4,
        "n_beat_bars": 4,
        "part_1_pitch": 55,
        "part_2_pitch": 72,
        "n_params": sum(p.numel() for p in model.parameters()),
        "val_loss": round(evals[-1]["val_loss"], 4) if evals else None,
        "start_ix": START_IX,
        # token id -> hit value, so the page can encode and decode exactly as training did
        "hits_tokens": {str(k): v for k, v in hits_vocab.items() if not isinstance(v, str)},
        "greedy_agreement": round(agree, 4),
        "n_examples": len(examples),
        "n_index": len(index),
    }

    for name, payload in [("meta", meta), ("examples", examples), ("index", index)]:
        path = os.path.join(data_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        print(f"  {name}.json  {os.path.getsize(path)/1e6:.2f} MB")

    total = sum(
        os.path.getsize(os.path.join(data_dir, f)) for f in os.listdir(data_dir)
    )
    print(f"\n{data_dir}: {total/1e6:.1f} MB total")


if __name__ == "__main__":
    main()
