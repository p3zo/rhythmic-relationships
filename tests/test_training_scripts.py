"""Checks over the training scripts themselves.

The backprop block is copy-pasted into every training script rather than shared, so these read
the source instead of calling it. They exist because the copies silently drifted once already:
`clip_grad_norm_` sat before `backward()` in all nine, which made every clip a no-op.
"""

import pathlib
import re

import torch
import torch.nn as nn

REPO_ROOT = pathlib.Path(__file__).parent.parent
MODELING_DIR = REPO_ROOT / "scripts" / "modeling"

CALLS = re.compile(
    r"\.backward\(\)|torch\.nn\.utils\.clip_grad_norm_\(|optimizer\.step\(\)"
)


def get_scripts_that_clip():
    return sorted(
        p
        for p in MODELING_DIR.glob("**/*.py")
        if "clip_grad_norm_" in p.read_text()
    )


def test_gradients_are_clipped_after_backward_and_before_step():
    scripts = get_scripts_that_clip()
    # Otherwise a glob that finds nothing would make the loop below vacuously pass
    assert scripts

    for path in scripts:
        calls = CALLS.findall(path.read_text())
        assert calls, f"{path}: no backprop calls found"

        # Clipping reads .grad, so it has to sit between the backward pass that fills it and the
        # step that consumes it
        for ix, call in enumerate(calls):
            if call != "torch.nn.utils.clip_grad_norm_(":
                continue
            rel = path.relative_to(REPO_ROOT)
            assert ix > 0 and calls[ix - 1] == ".backward()", (
                f"{rel}: clip_grad_norm_ is not preceded by backward(); "
                f"gradients are zero when it runs, so the clip does nothing"
            )
            assert ix + 1 < len(calls) and calls[ix + 1] == "optimizer.step()", (
                f"{rel}: clip_grad_norm_ is not followed by optimizer.step()"
            )


def test_clipping_before_backward_would_be_a_noop():
    """The failure mode the ordering test above protects against"""
    model = nn.Linear(4, 2)
    x = torch.randn(8, 4) * 1000

    def grad_norm_at_step(clip_first):
        model.zero_grad(set_to_none=True)
        loss = model(x).pow(2).mean()
        if clip_first:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            loss.backward()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        return torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item()

    assert grad_norm_at_step(clip_first=True) > 1.0
    assert grad_norm_at_step(clip_first=False) <= 0.5 + 1e-6
