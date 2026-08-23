"""Listen to and look at src/tgt/gen triplets from an eval inference directory.

The `hits` representation puts every note of a part on a single pitch, so a piano roll would be
one flat line: what carries the information is where the onsets land and how hard. The picture is
a step grid instead, and the audio plays the target take and then the model's take against the
same input part, so the two can be compared by ear in one listen.

Synthesis is a sine burst per onset rather than a soundfont, so this needs nothing installed
beyond the project dependencies.
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pretty_midi as pm
from scipy.io import wavfile

SAMPLE_RATE = 44100

# One sine burst per onset. The envelope is short and steep so onsets read as rhythm rather
# than as sustained tones piling up on a single pitch.
BURST_SECONDS = 0.18
DECAY = 25.0

ROLES = ["src", "tgt", "gen"]
ROW_LABELS = ["Melody (input)", "Bass (target)", "Bass (model)"]


def available_ixs(inference_dir, sampler):
    found = set()
    for path in glob.glob(os.path.join(inference_dir, f"*_{sampler}_gen.mid")):
        match = re.match(r"(\d+)_", os.path.basename(path))
        if match:
            found.add(int(match.group(1)))
    return sorted(found)


def read_steps(path, n_steps, step_seconds):
    """Velocity per grid step, as `get_pmid_from_hits` laid it out."""
    steps = np.zeros(n_steps)
    notes = pm.PrettyMIDI(path).instruments[0].notes
    for note in notes:
        step = round(note.start / step_seconds)
        if step >= n_steps:
            raise ValueError(
                f"{path} has an onset at step {step}, past the {n_steps}-step grid. "
                "Pass --n_steps to match the config's sequence_len / block_size."
            )
        steps[step] = note.velocity
    return steps, notes


def synthesize(notes, n_samples):
    track = np.zeros(n_samples)
    for note in notes:
        start = int(note.start * SAMPLE_RATE)
        length = min(int(BURST_SECONDS * SAMPLE_RATE), n_samples - start)
        if length <= 0:
            continue
        t = np.arange(length) / SAMPLE_RATE
        freq = pm.note_number_to_hz(note.pitch)
        track[start : start + length] += (
            np.sin(2 * np.pi * freq * t) * np.exp(-DECAY * t) * (note.velocity / 127)
        )
    return track


def draw(ax, steps, ix, sampler, n_steps, resolution, n_beat_bars):
    grid = np.stack([steps["src"], steps["tgt"], steps["gen"]])
    # Onsets as ink on paper: white is a rest, darkness is velocity
    ax.imshow(grid, cmap="Greys", vmin=0, vmax=127, aspect="auto", interpolation="nearest")

    for step in range(n_steps + 1):
        if step % (resolution * n_beat_bars) == 0:
            ax.axvline(step - 0.5, color="#1f77b4", lw=2.0)
        elif step % resolution == 0:
            ax.axvline(step - 0.5, color="#999999", lw=0.7)
    for row in (0.5, 1.5):
        ax.axhline(row, color="#1f77b4", lw=2.0)

    # Outline where the model put an onset the target did not, and vice versa
    for step in range(n_steps):
        if (steps["tgt"][step] > 0) != (steps["gen"][step] > 0):
            ax.add_patch(
                plt.Rectangle(
                    (step - 0.42, 1.58), 0.84, 0.84, fill=False,
                    edgecolor="#d62728", lw=2.0, zorder=5,
                )
            )

    agree = int(((steps["tgt"] > 0) == (steps["gen"] > 0)).sum())
    ax.set_yticks(range(3), ROW_LABELS, fontsize=8)
    ax.set_ylabel(
        f"ix {ix}\n{int((steps['tgt'] > 0).sum())} vs {int((steps['gen'] > 0).sum())} onsets"
        f"\n{agree}/{n_steps} steps agree",
        fontsize=7.5, rotation=0, ha="right", va="center", labelpad=28,
    )
    return agree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_dir", type=str, required=True)
    parser.add_argument("--ix", type=str, default="0", help="comma-separated, or 'all'")
    parser.add_argument("--limit", type=int, default=8, help="max examples when --ix all")
    parser.add_argument("--sampler", type=str, default="nucleus")
    parser.add_argument("--n_steps", type=int, default=32, help="sequence_len / block_size")
    parser.add_argument("--resolution", type=int, default=4, help="subdivisions per beat")
    parser.add_argument("--n_beat_bars", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2, help="loops of each take")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--play", action="store_true", help="afplay the result when done")
    args = parser.parse_args()

    # `get_pmid_from_hits` writes one step per `1 / (resolution * 2)` seconds
    step_seconds = 1 / (args.resolution * 2)
    seg_seconds = args.n_steps * step_seconds

    found = available_ixs(args.inference_dir, args.sampler)
    if not found:
        raise SystemExit(
            f"No *_{args.sampler}_gen.mid in {args.inference_dir}. "
            "Check the sampler name and that an eval has run."
        )
    if args.ix == "all":
        ixs = found[: args.limit]
    else:
        ixs = [int(i) for i in args.ix.split(",")]
        unknown = [i for i in ixs if i not in found]
        if unknown:
            raise SystemExit(f"No generation for ix {unknown}. Available: {found}")

    outdir = args.outdir or os.path.join(os.path.dirname(args.inference_dir.rstrip("/")), "audition")
    os.makedirs(outdir, exist_ok=True)
    label = args.ix if args.ix != "all" else f"first{len(ixs)}"
    stem = os.path.join(outdir, f"{label.replace(',', '-')}_{args.sampler}")

    n_samples = int(seg_seconds * SAMPLE_RATE)
    short_gap = np.zeros((int(0.4 * SAMPLE_RATE), 2))
    long_gap = np.zeros((int(1.0 * SAMPLE_RATE), 2))

    fig, axes = plt.subplots(
        len(ixs), 1,
        figsize=(min(2 + args.n_steps * 0.32, 22), 1.9 * len(ixs) + 0.7),
        squeeze=False,
    )

    pieces = []
    for row, ix in enumerate(ixs):
        steps, notes = {}, {}
        for role in ROLES:
            path = os.path.join(args.inference_dir, f"{ix}_{args.sampler}_{role}.mid")
            if not os.path.isfile(path):
                raise SystemExit(f"Missing {path}")
            steps[role], notes[role] = read_steps(path, args.n_steps, step_seconds)

        # One file with the three parts as three tracks, so a DAW import is a single drag
        merged = pm.PrettyMIDI()
        for role, row_label in zip(ROLES, ROW_LABELS):
            source = pm.PrettyMIDI(
                os.path.join(args.inference_dir, f"{ix}_{args.sampler}_{role}.mid")
            ).instruments[0]
            merged.instruments.append(
                pm.Instrument(program=source.program, is_drum=source.is_drum,
                              name=row_label)
            )
            merged.instruments[-1].notes = list(source.notes)
        merged_path = os.path.join(outdir, f"{ix}_{args.sampler}_3track.mid")
        merged.write(merged_path)

        tracks = {r: synthesize(notes[r], n_samples) for r in ROLES}
        # Input part left, bass right, so the two stay tellable apart
        with_target = np.stack([tracks["src"], tracks["tgt"]], axis=-1)
        with_model = np.stack([tracks["src"], tracks["gen"]], axis=-1)
        pieces += [
            np.tile(with_target, (args.repeats, 1)), short_gap,
            np.tile(with_model, (args.repeats, 1)), long_gap,
        ]

        agree = draw(axes[row][0], steps, ix, args.sampler,
                     args.n_steps, args.resolution, args.n_beat_bars)
        print(f"ix {ix:>3}  target/model onsets "
              f"{int((steps['tgt'] > 0).sum()):>2}/{int((steps['gen'] > 0).sum()):>2}  "
              f"step agreement {agree}/{args.n_steps}")

    for row in range(len(ixs) - 1):
        axes[row][0].set_xticks([])
    last = axes[-1][0]
    last.set_xticks(range(0, args.n_steps, args.resolution),
                    [str(s // args.resolution + 1) for s in range(0, args.n_steps, args.resolution)])
    last.set_xlabel("beat")
    axes[0][0].set_title(
        f"{args.inference_dir}\nsampler={args.sampler}  "
        "(red = model and target disagree on that step)", fontsize=9)

    fig.tight_layout()
    png_path = f"{stem}.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)

    audio = np.concatenate(pieces)
    peak = np.abs(audio).max()
    if peak == 0:
        raise SystemExit("Every part is silent; nothing to hear.")
    audio = audio / peak * 0.9
    wav_path = f"{stem}.wav"
    wavfile.write(wav_path, SAMPLE_RATE, (audio * 32767).astype(np.int16))

    print(f"\nSaved {png_path}")
    print(f"Saved {wav_path}  ({audio.shape[0] / SAMPLE_RATE:.1f}s)")
    print(f"  per example: input+target x{args.repeats}, pause, input+model x{args.repeats}")
    print("  input in the left channel, bass in the right")
    print(f"Saved {len(ixs)} x *_3track.mid in {outdir}")
    print("  three named tracks per file - drag one into GarageBand to hear it with real sounds")

    if args.play:
        os.system(f'afplay "{wav_path}"')


if __name__ == "__main__":
    main()
