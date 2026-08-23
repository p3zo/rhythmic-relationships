"""Local web interface for playing with a trained hits encoder-decoder.

Draw or load an input part, generate the model's answer, hear both, and see how the answer
relates to what it was given. Serves on localhost with the standard library only.

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

import numpy as np
import torch
from rhythmic_relationships import DATASETS_DIR, MODELS_DIR, REPRESENTATIONS_DIRNAME
from rhythmic_relationships import REPRESENTATIONS_FILENAME
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

STATE = {}


def load_examples(dataset_name, n_examples, part):
    """Real input parts from the dataset, for hearing what the model does with genuine material."""
    dataset_dir = os.path.join(DATASETS_DIR, dataset_name)
    reprs_path = os.path.join(dataset_dir, REPRESENTATIONS_FILENAME)
    if not os.path.isfile(reprs_path):
        raise SystemExit(
            f"No dataset at {dataset_dir}, which is where the model's config points.\n"
            f"Prepare it first, or pass --examples 0 to run without real inputs."
        )

    with open(reprs_path, "r") as f:
        hits_ix = f.read().split(",").index("hits")

    paths = glob.glob(
        os.path.join(dataset_dir, REPRESENTATIONS_DIRNAME, "**", "*.npz"), recursive=True
    )
    random.shuffle(paths)

    examples = []
    for path in paths:
        with np.load(path, allow_pickle=True) as npz:
            keys = [k for k in npz.files if k.endswith(f"_{part}")]
            for key in keys:
                examples.append(
                    {
                        "hits": [round(float(v), 4) for v in npz[key][0][hits_ix]],
                        "name": f"{os.path.basename(path)[:-4]} seg {key.split('_')[0]}",
                    }
                )
                if len(examples) >= n_examples:
                    return examples
    return examples


def describe(hits, pitch, resolution):
    roll = get_roll_from_hits([h * 127 for h in hits], pitch=pitch, resolution=resolution)
    descriptors = pianoroll2descriptors(roll, resolution, drums=False)
    return {k: descriptors[k] for k in DESCRIPTORS}


def generate(hits, sampler, temperature, nucleus_p):
    model, config, device = STATE["model"], STATE["config"], STATE["device"]
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
    together = sorted(set(in_onsets) & set(out_onsets))

    return {
        "gen_hits": [round(float(v), 4) for v in gen_hits],
        "onset_probs": [round(float(p), 4) for p in onset_probs],
        "in_onsets": in_onsets,
        "out_onsets": out_onsets,
        "together": together,
        "in_gaps": sorted(set(out_onsets) - set(in_onsets)),
        "descriptors": {
            "input": describe(hits, STATE["part_1_pitch"], resolution),
            "output": describe(gen_hits, STATE["part_2_pitch"], resolution),
        },
    }


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # the request log drowns out the startup banner

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(PAGE_PATH, "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/meta":
            config = STATE["config"]
            body = json.dumps(
                {
                    "model_path": STATE["model_path"],
                    "n_steps": config["model"]["context_len"],
                    "resolution": STATE["resolution"],
                    "n_beat_bars": STATE["n_beat_bars"],
                    "part_1": config["data"]["part_1"],
                    "part_2": config["data"]["part_2"],
                    "part_1_pitch": STATE["part_1_pitch"],
                    "part_2_pitch": STATE["part_2_pitch"],
                    "dataset_name": config["data"]["dataset_name"],
                    "n_params": STATE["n_params"],
                    "final_eval": STATE["final_eval"],
                    "velocity_levels": STATE["velocity_levels"],
                    "n_examples": len(STATE["examples"]),
                    "descriptors": DESCRIPTORS,
                }
            ).encode()
            self._send(200, body, "application/json")
        elif self.path == "/example":
            if not STATE["examples"]:
                self._send(404, b'{"error":"no dataset examples loaded"}', "application/json")
                return
            body = json.dumps(random.choice(STATE["examples"])).encode()
            self._send(200, body, "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/generate":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length))
        try:
            result = generate(
                hits=payload["hits"],
                sampler=payload.get("sampler", "nucleus"),
                temperature=float(payload.get("temperature", 1.0)),
                nucleus_p=float(payload.get("nucleus_p", 0.92)),
            )
        except (ValueError, KeyError) as e:
            self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")
            return
        self._send(200, json.dumps(result).encode(), "application/json")


def newest_model():
    paths = glob.glob(os.path.join(MODELS_DIR, "hits_encdec", "**", "model.pt"), recursive=True)
    if not paths:
        raise SystemExit(
            f"No hits_encdec model.pt under {os.path.join(MODELS_DIR, 'hits_encdec')}. "
            "Train one first, or pass --model_path."
        )
    return max(paths, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="default: newest hits_encdec")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--examples", type=int, default=64, help="real inputs to load; 0 to skip")
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--n_beat_bars", type=int, default=4)
    # The pitches the eval writes each part on, so the descriptors match what it reports
    parser.add_argument("--part_1_pitch", type=int, default=55)
    parser.add_argument("--part_2_pitch", type=int, default=72)
    args = parser.parse_args()

    model_path = args.model_path or newest_model()
    model, config = load_model(model_path, TransformerEncoderDecoder)
    model.eval()
    model.to(args.device)

    if config["data"]["block_size"] != 1:
        raise SystemExit(
            f"This interface reads one token per step, but the model was trained with "
            f"block_size={config['data']['block_size']}."
        )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    evals = checkpoint.get("evals") or []

    STATE.update(
        model=model,
        config=config,
        device=args.device,
        model_path=model_path,
        n_params=sum(p.numel() for p in model.parameters()),
        resolution=args.resolution,
        n_beat_bars=args.n_beat_bars,
        part_1_pitch=args.part_1_pitch,
        part_2_pitch=args.part_2_pitch,
        velocity_levels=[v for v in get_hits_vocab().values() if isinstance(v, float) or v == 0],
        final_eval={
            k: round(evals[-1][k], 4)
            for k in ("val_loss", "train_loss", "val_train_loss_pct_diff")
            if evals and k in evals[-1]
        },
        examples=[],
    )

    if args.examples:
        print(f"Loading {args.examples} real {config['data']['part_1']} inputs...")
        STATE["examples"] = load_examples(
            config["data"]["dataset_name"], args.examples, config["data"]["part_1"]
        )

    print(f"\n  model      {model_path}")
    print(f"  {config['data']['part_1']} -> {config['data']['part_2']}, "
          f"{config['model']['context_len']} steps, {STATE['n_params']:,} params")
    print(f"  final eval {STATE['final_eval']}")
    print(f"  examples   {len(STATE['examples'])}")
    print(f"\n  http://localhost:{args.port}\n")

    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
