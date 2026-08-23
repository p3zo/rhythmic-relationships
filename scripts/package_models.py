"""Bundle trained models into an archive to attach to a GitHub release.

Weights are the one output of this repo that cannot be regenerated: a run takes hours and the
dataset shuffle that produced it is not reproducible across machines. They also do not belong in
git, where every version would be kept forever and none of them would delta-compress. So they
ship as release assets instead, under a version of their own that is bumped whenever the set of
models changes.

Two archives, because they answer different questions. The weights archive is what you need to
run a model, and stays small. The checkpoints archive carries the optimizer state as well, which
is three times the size and only matters if you intend to resume training a run.
"""

import argparse
import hashlib
import json
import os
import subprocess
import zipfile

import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(model_path):
    weights = torch.load(model_path, map_location="cpu", weights_only=False)
    config = weights["config"]
    evals = weights.get("evals") or []
    return {
        "run": os.path.basename(os.path.dirname(model_path)),
        "part_1": config["data"]["part_1"],
        "part_2": config["data"]["part_2"],
        "model_class": weights["model_class"],
        "dataset": config["data"]["dataset_name"],
        "n_steps": config["model"]["context_len"],
        "n_params": weights["n_params"],
        # Evals run several times an epoch, so this is not an epoch count; the epoch comes from
        # the checkpoint, which is the only place it is recorded
        "n_evals": len(evals),
        "val_loss": round(evals[-1]["val_loss"], 4) if evals else None,
        "bytes": os.path.getsize(model_path),
        "sha256": sha256(model_path),
    }


def latest_checkpoint(model_dir):
    """The checkpoint training left behind, if any. Numbered by epoch, earlier ones deleted."""
    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    if not os.path.isdir(checkpoints_dir):
        return None, None
    epochs = sorted(int(f) for f in os.listdir(checkpoints_dir))
    if not epochs:
        return None, None
    return epochs[-1], os.path.join(checkpoints_dir, str(epochs[-1]))


def write_zip(path, members, manifest):
    # Stored rather than deflated: these are float32 tensors, so compression buys a few percent
    # for minutes of CPU
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for arcname, source in members:
            z.write(source, arcname)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
    print(f"{path}  {os.path.getsize(path)/1e6:.1f} MB  ({len(members)} models)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, nargs="+")
    parser.add_argument("--version", type=str, required=True,
                        help="Semantic version for this set of models, e.g. 1.0.0")
    parser.add_argument("--outdir", type=str, default=os.path.join("output", "releases"))
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()

    weights, checkpoints, described = [], [], []
    for model_path in args.model_path:
        entry = describe(model_path)
        name = f"{entry['part_1']}_{entry['part_2']}_{entry['run']}"
        weights.append((f"{name}/model.pt", model_path))

        epoch, checkpoint_path = latest_checkpoint(os.path.dirname(model_path))
        entry["epochs"] = epoch
        if checkpoint_path:
            # A checkpoint from an earlier epoch than the shipped weights would resume a
            # different model than the one in the other archive, which is worth refusing
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            shipped = torch.load(model_path, map_location="cpu", weights_only=False)
            if not all(torch.equal(checkpoint["model_state_dict"][k], v)
                       for k, v in shipped["model_state_dict"].items()):
                raise ValueError(
                    f"{entry['run']}: checkpoint at epoch {epoch} holds different weights than "
                    f"model.pt, so it cannot resume the model being released"
                )
            checkpoints.append((f"{name}/checkpoints/{epoch}", checkpoint_path))
        described.append(entry)
        print(f"{entry['part_1']:>8} -> {entry['part_2']:<8} {entry['run']:<24} "
              f"{epoch} epochs, val {entry['val_loss']}")

    manifest = {"version": args.version, "commit": commit, "models": described}
    print()
    write_zip(os.path.join(args.outdir, f"hits-encdec-models-{args.version}.zip"),
              weights, manifest)
    if checkpoints:
        write_zip(os.path.join(args.outdir, f"hits-encdec-checkpoints-{args.version}.zip"),
                  checkpoints, manifest)

    manifest_path = os.path.join(args.outdir, f"MANIFEST-{args.version}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"{manifest_path}")


if __name__ == "__main__":
    main()
