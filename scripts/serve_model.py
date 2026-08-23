"""Local web interface for playing with trained hits encoder-decoder models.

Pick a model, draw or load an input part, generate the model's answer, hear both, and see how the
answer relates to what it was given. Serves on localhost with the standard library only.

Once a rhythm has been generated, the dataset is searched for the real segments whose rhythm
is closest to it. Playing one of those against the input is a mashup: your melody with a bassline
from another song that happens to move the way the model said it should.

The interesting number here is the per-step onset probability. Generation samples one sequence,
but the model's actual opinion is a distribution, so after sampling we run the generated sequence
back through in one teacher-forced pass. The model is causal, so the logits at step t are
conditioned on exactly the tokens that were in hand when step t was sampled - the same
distribution the sampler drew from, not an approximation of it.
"""

import argparse
import glob
import http.server
import json
import os
import random
import urllib.parse

import numpy as np
import torch
from rhythmic_relationships import (
    DATASETS_DIR,
    MODELS_DIR,
    REPRESENTATIONS_DIRNAME,
    REPRESENTATIONS_FILENAME,
)
from rhythmic_relationships.data import get_hits_from_hits_seq, tokenize_hits
from rhythmic_relationships.evaluate import hits_inference
from rhythmic_relationships.io import get_roll_from_hits
from rhythmic_relationships.model_utils import load_model
from rhythmic_relationships.models.hits_encdec import TransformerEncoderDecoder
from rhythmic_relationships.vocab import START_IX, get_hits_vocab
from rhythmtoolbox import pianoroll2descriptors

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PAGE_PATH = os.path.join(THIS_DIR, "serve_model.html")

# The single-step token standing for "no hit"
REST_IX = next(ix for ix, value in get_hits_vocab().items() if value == 0)

# Only these carry a value for a part on one pitch; the rest need a real pitch spread
DESCRIPTORS = ["stepDensity", "sync", "syness", "balance", "evenness"]

STATE = {"models": {}, "loaded": {}, "examples": {}, "indexes": {}}


def discover_models():
    """Every trained hits_encdec run, as `run_train` lays them out: <part_1>_<part_2>/<run>.

    Each checkpoint is opened once here so the dropdown can show what a run actually is - which
    parts, which dataset, what it scored - rather than making the choice from a directory name.
    """
    root = os.path.join(MODELS_DIR, "hits_encdec")
    paths = sorted(glob.glob(os.path.join(root, "*", "*", "model.pt")))
    if not paths:
        raise SystemExit(
            f"No hits_encdec model.pt under {root}. Train one first with:\n"
            "  uv run python scripts/modeling/run_train.py --model hits_encdec"
        )

    print(f"  reading {len(paths)} checkpoint(s)...")
    models = {}
    for path in paths:
        run = os.path.basename(os.path.dirname(path))
        pair = os.path.basename(os.path.dirname(os.path.dirname(path)))
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        evals = checkpoint.get("evals") or []
        val_loss = evals[-1].get("val_loss") if evals else None
        # A select option collapses runs of whitespace, so separate the fields explicitly
        label = " · ".join([
            f"{config['data']['part_1']} → {config['data']['part_2']}",
            f"val {val_loss:.4f}" if val_loss is not None else "val ?",
            config["data"]["dataset_name"],
            run,
        ])
        models[f"{pair}/{run}"] = {
            "path": path,
            "label": label,
            "val_loss": val_loss,
            "mtime": os.path.getmtime(path),
        }
    return models


def get_model(model_id):
    """Load a model on first use and keep it; they are a few MB each."""
    if model_id not in STATE["models"]:
        raise KeyError(f"Unknown model {model_id}")
    if model_id not in STATE["loaded"]:
        entry = STATE["models"][model_id]
        model, config = load_model(entry["path"], TransformerEncoderDecoder)
        model.eval()
        model.to(STATE["device"])
        if config["data"]["block_size"] != 1:
            raise ValueError(
                f"{model_id} was trained with block_size={config['data']['block_size']}, "
                "but this interface reads one token per step."
            )
        checkpoint = torch.load(entry["path"], map_location="cpu", weights_only=False)
        evals = checkpoint.get("evals") or []
        STATE["loaded"][model_id] = {
            "model": model,
            "config": config,
            "n_params": sum(p.numel() for p in model.parameters()),
            "final_eval": {
                k: round(evals[-1][k], 4)
                for k in ("val_loss", "train_loss", "val_train_loss_pct_diff")
                if evals and k in evals[-1]
            },
        }
    return STATE["loaded"][model_id]


def load_examples(dataset_name, part, n_examples):
    """Real segments of one part: the hits the model reads, and the pitches they were played at."""
    key = (dataset_name, part)
    if key in STATE["examples"]:
        return STATE["examples"][key]

    dataset_dir = os.path.join(DATASETS_DIR, dataset_name)
    if not os.path.isfile(os.path.join(dataset_dir, REPRESENTATIONS_FILENAME)):
        raise FileNotFoundError(
            f"No dataset at {dataset_dir}, which is where this model's config points. "
            "Prepare it, or start the server with --examples 0."
        )
    hits_ix, onset_roll_ix = read_reprs(dataset_dir, ["hits", "onset_roll"])

    paths = glob.glob(
        os.path.join(dataset_dir, REPRESENTATIONS_DIRNAME, "**", "*.npz"), recursive=True
    )
    random.shuffle(paths)

    examples = []
    for path in paths:
        with np.load(path, allow_pickle=True) as npz:
            for npz_key in [k for k in npz.files if k.endswith(f"_{part}")]:
                reprs = npz[npz_key][0]
                examples.append(
                    {
                        "hits": [round(float(v), 4) for v in reprs[hits_ix]],
                        # Same onsets as `hits`, but keeping the pitch each was played at
                        "pitched": pitched_from_onset_roll(np.asarray(reprs[onset_roll_ix])),
                        "name": f"{os.path.basename(path)[:-4]} seg {npz_key.split('_')[0]}",
                    }
                )
                if len(examples) >= n_examples:
                    STATE["examples"][key] = examples
                    return examples
    STATE["examples"][key] = examples
    return examples


def read_reprs(dataset_dir, names):
    """The representation index for each name, from the dataset's own manifest."""
    with open(os.path.join(dataset_dir, REPRESENTATIONS_FILENAME), "r") as f:
        manifest = f.read().split(",")
    return [manifest.index(n) for n in names]


def pitched_from_onset_roll(onset_roll):
    """(step, pitch, velocity) for every onset, keeping the pitch it was played at."""
    steps, pitches = np.nonzero(onset_roll)
    return [
        [int(s), int(p), round(float(onset_roll[s, p]), 4)] for s, p in zip(steps, pitches)
    ]


def build_index(dataset_name, part, n_segments):
    """Onset vectors for `n_segments` real segments of one part, for the nearest-rhythm search.

    Only the vectors and a locator per segment are held. The pitches are read back from the npz
    for the handful of segments that come out of a search, which keeps this to a few MB.
    """
    key = (dataset_name, part, n_segments)
    if key in STATE["indexes"]:
        return STATE["indexes"][key]

    dataset_dir = os.path.join(DATASETS_DIR, dataset_name)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"No dataset at {dataset_dir}")
    (hits_ix,) = read_reprs(dataset_dir, ["hits"])

    paths = glob.glob(
        os.path.join(dataset_dir, REPRESENTATIONS_DIRNAME, "**", "*.npz"), recursive=True
    )
    random.shuffle(paths)

    vectors, locators, used_paths = [], [], []
    for path in paths:
        with np.load(path, allow_pickle=True) as npz:
            npz_keys = [k for k in npz.files if k.endswith(f"_{part}")]
            if not npz_keys:
                continue
            path_ix = len(used_paths)
            used_paths.append(path)
            for npz_key in npz_keys:
                vectors.append(np.asarray(npz[npz_key][0][hits_ix], dtype=np.float32))
                locators.append((path_ix, npz_key))
        if len(vectors) >= n_segments:
            break

    if not vectors:
        raise FileNotFoundError(f"No {part} segments in {dataset_dir}")

    index = {
        "hits": np.stack(vectors),
        "locators": locators,
        "paths": used_paths,
    }
    STATE["indexes"][key] = index
    print(
        f"  indexed {len(locators):,} {part} segments from {len(used_paths):,} files "
        f"({index['hits'].nbytes / 1e6:.1f} MB)"
    )
    return index


def nearest_rhythms(dataset_name, part, target_hits, k):
    """The `k` closest real segments to `target_hits`, one per source file.

    Ranked by how many of the 32 steps disagree about having an onset, since that is what "the
    same rhythm" means here, and ties broken by how close the velocities are. Capped at one hit
    per file so ten results are ten different pieces of music rather than ten takes of one.
    """
    index = build_index(dataset_name, part, STATE["n_index"])
    target = np.asarray(target_hits, dtype=np.float32)

    onset_disagreement = ((index["hits"] > 0) != (target > 0)).sum(axis=1)
    velocity_distance = np.abs(index["hits"] - target).sum(axis=1)
    order = np.lexsort((velocity_distance, onset_disagreement))

    dataset_dir = os.path.join(DATASETS_DIR, dataset_name)
    (onset_roll_ix,) = read_reprs(dataset_dir, ["onset_roll"])

    results, seen_files = [], set()
    for ix in order:
        path_ix, npz_key = index["locators"][ix]
        path = index["paths"][path_ix]
        if path in seen_files:
            continue
        seen_files.add(path)
        with np.load(path, allow_pickle=True) as npz:
            onset_roll = np.asarray(npz[npz_key][0][onset_roll_ix])
        results.append(
            {
                "name": f"{os.path.basename(path)[:-4]} seg {npz_key.split('_')[0]}",
                "hits": [round(float(v), 4) for v in index["hits"][ix]],
                "pitched": pitched_from_onset_roll(onset_roll),
                "steps_apart": int(onset_disagreement[ix]),
            }
        )
        if len(results) >= k:
            break
    return results


def describe(hits, pitch, resolution):
    roll = get_roll_from_hits([h * 127 for h in hits], pitch=pitch, resolution=resolution)
    descriptors = pianoroll2descriptors(roll, resolution, drums=False)
    return {k: descriptors[k] for k in DESCRIPTORS}


def generate(model_id, hits, sampler, temperature, nucleus_p):
    entry = get_model(model_id)
    model, config = entry["model"], entry["config"]
    device = STATE["device"]
    n_steps = config["model"]["context_len"]

    if len(hits) != n_steps:
        raise ValueError(f"Expected {n_steps} steps, got {len(hits)}")

    src = torch.tensor(
        [tokenize_hits(np.array(hits), block_size=config["data"]["block_size"])],
        dtype=torch.long,
        device=device,
    )

    seq = hits_inference(
        model=model,
        src=src,
        n_tokens=n_steps,
        temperature=temperature,
        device=device,
        sampler=sampler,
        nucleus_p=nucleus_p,
    )

    gen_hits = get_hits_from_hits_seq(
        seq.cpu().numpy(),
        part=config["data"]["part_2"],
        block_size=config["data"]["block_size"],
    )

    # Replay the sampled sequence to recover the distribution each step was drawn from
    shifted = torch.roll(seq.unsqueeze(0), 1, dims=1)
    shifted[:, 0] = START_IX
    with torch.no_grad():
        logits = model(src=src, tgt=shifted)
    probs = torch.softmax(logits[0].float(), dim=-1)
    onset_probs = (1 - probs[:, REST_IX]).tolist()

    resolution = STATE["resolution"]
    in_onsets = [i for i, v in enumerate(hits) if v > 0]
    out_onsets = [i for i, v in enumerate(gen_hits) if v > 0]

    return {
        "gen_hits": [round(float(v), 4) for v in gen_hits],
        "onset_probs": [round(float(p), 4) for p in onset_probs],
        "in_onsets": in_onsets,
        "out_onsets": out_onsets,
        "together": sorted(set(in_onsets) & set(out_onsets)),
        "in_gaps": sorted(set(out_onsets) - set(in_onsets)),
        "descriptors": {
            "input": describe(hits, STATE["part_1_pitch"], resolution),
            "output": describe(gen_hits, STATE["part_2_pitch"], resolution),
        },
    }


def meta_for(model_id):
    entry = get_model(model_id)
    config = entry["config"]
    return {
        "model_id": model_id,
        "models": [
            {"id": k, "label": v["label"]}
            for k, v in sorted(
                STATE["models"].items(),
                key=lambda kv: (kv[1]["val_loss"] is None, kv[1]["val_loss"] or 0),
            )
        ],
        "n_steps": config["model"]["context_len"],
        "resolution": STATE["resolution"],
        "n_beat_bars": STATE["n_beat_bars"],
        "part_1": config["data"]["part_1"],
        "part_2": config["data"]["part_2"],
        "part_1_pitch": STATE["part_1_pitch"],
        "part_2_pitch": STATE["part_2_pitch"],
        "dataset_name": config["data"]["dataset_name"],
        "n_params": entry["n_params"],
        "final_eval": entry["final_eval"],
        "has_examples": STATE["n_examples"] > 0,
        "n_index": STATE["n_index"],
        "descriptors": DESCRIPTORS,
    }


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def log_message(self, fmt, *args):
        pass  # the request log drowns out the startup banner

    def _model_id(self, query):
        return query.get("model", [STATE["default_model"]])[0]

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)

        if parsed.path in ("/", "/index.html"):
            with open(PAGE_PATH, "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
            return

        try:
            if parsed.path == "/meta":
                self._json(200, meta_for(self._model_id(query)))
                return
            if parsed.path == "/example":
                if not STATE["n_examples"]:
                    self._json(404, {"error": "started with --examples 0"})
                    return
                entry = get_model(self._model_id(query))
                examples = load_examples(
                    entry["config"]["data"]["dataset_name"],
                    entry["config"]["data"]["part_1"],
                    STATE["n_examples"],
                )
                if not examples:
                    self._json(404, {"error": "no segments found for this part"})
                    return
                self._json(200, random.choice(examples))
                return
        except (KeyError, ValueError, FileNotFoundError) as e:
            self._json(400, {"error": str(e)})
            return

        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length)) if length else {}
        model_id = payload.get("model") or STATE["default_model"]

        try:
            if path == "/generate":
                self._json(
                    200,
                    generate(
                        model_id=model_id,
                        hits=payload["hits"],
                        sampler=payload.get("sampler", "nucleus"),
                        temperature=float(payload.get("temperature", 1.0)),
                        nucleus_p=float(payload.get("nucleus_p", 0.92)),
                    ),
                )
                return
            if path == "/neighbors":
                config = get_model(model_id)["config"]
                self._json(
                    200,
                    {
                        "part": config["data"]["part_2"],
                        "matches": nearest_rhythms(
                            dataset_name=config["data"]["dataset_name"],
                            part=config["data"]["part_2"],
                            target_hits=payload["hits"],
                            k=int(payload.get("k", 10)),
                        ),
                    },
                )
                return
        except (ValueError, KeyError, FileNotFoundError) as e:
            self._json(400, {"error": str(e)})
            return

        self._send(404, b"not found", "text/plain")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="default: newest run")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--examples", type=int, default=64, help="real inputs to load; 0 to skip")
    parser.add_argument(
        "--index",
        type=int,
        default=100000,
        help="segments to index for the nearest-rhythm search",
    )
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--n_beat_bars", type=int, default=4)
    # The pitches the eval writes each part on, so the descriptors match what it reports
    parser.add_argument("--part_1_pitch", type=int, default=55)
    parser.add_argument("--part_2_pitch", type=int, default=72)
    args = parser.parse_args()

    STATE["models"] = discover_models()
    STATE.update(
        device=args.device,
        resolution=args.resolution,
        n_beat_bars=args.n_beat_bars,
        part_1_pitch=args.part_1_pitch,
        part_2_pitch=args.part_2_pitch,
        n_examples=args.examples,
        n_index=args.index,
    )

    if args.model_path:
        wanted = os.path.abspath(args.model_path)
        matches = [k for k, v in STATE["models"].items() if os.path.abspath(v["path"]) == wanted]
        if not matches:
            raise SystemExit(f"{args.model_path} is not one of: {sorted(STATE['models'])}")
        STATE["default_model"] = matches[0]
    else:
        # Lowest val loss, not newest: the newest run is as likely to be a smoke test
        STATE["default_model"] = min(
            STATE["models"],
            key=lambda k: (
                STATE["models"][k]["val_loss"] is None,
                STATE["models"][k]["val_loss"] or 0,
            ),
        )

    print(f"\n  {len(STATE['models'])} model(s):")
    for model_id, entry in sorted(
        STATE["models"].items(),
        key=lambda kv: (kv[1]["val_loss"] is None, kv[1]["val_loss"] or 0),
    ):
        marker = "*" if model_id == STATE["default_model"] else " "
        print(f"   {marker} {entry['label']}")
    print(f"\n  http://localhost:{args.port}\n")

    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
