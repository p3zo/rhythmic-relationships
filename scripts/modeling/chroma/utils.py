import torch


def parse_batch(batch, device):
    yb = batch
    yb_shifted = torch.roll(yb, 1)
    yb_shifted[:, 0] = torch.zeros((yb.shape[0],))
    return yb.to(device), yb_shifted.to(device)


def compute_loss(logits, y, loss_fn):
    B, T, C = logits.shape
    return loss_fn(logits.view(B * T, C), y.view(y.shape[0] * y.shape[1]))
