import torch
import torch.nn as nn
from x_transformers import TransformerWrapper, Decoder


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        n_layer,
        n_head,
        n_embed,
        dropout,
        vocab_size,
        sequence_len,
    ):
        super().__init__()

        self.decoder = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=sequence_len,
            attn_layers=Decoder(
                dim=n_embed,
                depth=n_layer,
                heads=n_head,
                layer_dropout=dropout,
            ),
        )

    def forward(self, x):
        return self.decoder(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get the predictions
            logits = self(idx)

            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)

            # apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)  # (B, C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx
