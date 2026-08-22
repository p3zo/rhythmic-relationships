import torch

from rhythmic_relationships.vocab import START_IX


def parse_batch(batch, device):
    yb = batch

    # Teacher forcing: the decoder reads the target shifted right by one, seeded with `start`
    yb_shifted = torch.roll(yb, 1, dims=1)
    yb_shifted[:, 0] = START_IX

    return yb.to(device), yb_shifted.to(device)


def compute_loss(logits, y, loss_fn):
    B, T, C = logits.shape
    return loss_fn(logits.view(B * T, C), y.view(y.shape[0] * y.shape[1]))
