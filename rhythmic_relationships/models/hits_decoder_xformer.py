import torch
import torch.nn as nn
from x_transformers import TransformerWrapper, Decoder

from rhythmic_relationships.models.hits_encdec import get_causal_mask


class HitsDecoder(nn.Module):
    def __init__(
        self,
        n_layer,
        n_head,
        d_model,
        d_ff,
        dropout,
        vocab_size,
        context_len,
        pad_ix,
    ):
        super().__init__()

        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.pad_ix = pad_ix

        self.decoder = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=context_len,
            l2norm_embed=True,
            attn_layers=Decoder(
                dim=d_model,
                depth=n_layer,
                heads=n_head,
                layer_dropout=dropout,
                rotary_pos_emb=True,
                ff_glu=True,
                ff_no_bias=True,
                # x-transformers sizes the feed-forward as a multiple of the model dim
                ff_mult=d_ff / d_model,
            ),
        )

    def forward(self, x):
        attn_mask = get_causal_mask(x.size(1), device=x.device, boolean=True)
        out = self.decoder(x, attn_mask=attn_mask)
        return out
