import torch
import torch.nn as nn
from x_transformers import TransformerWrapper, Encoder, Decoder

from rhythmic_relationships.models.hits_decoder import get_causal_mask


class TransformerEncoderDecoder(nn.Module):
    def __init__(
        self,
        context_len,
        src_vocab_size,
        tgt_vocab_size,
        enc_n_embed,
        enc_n_layer,
        enc_n_head,
        enc_dropout,
        dec_n_embed,
        dec_n_layer,
        dec_n_head,
        dec_dropout,
        pad_ix,
    ):
        super().__init__()

        self.context_len = context_len
        self.pad_ix = pad_ix

        self.encoder = TransformerWrapper(
            num_tokens=src_vocab_size,
            max_seq_len=context_len,
            l2norm_embed=True,
            attn_layers=Encoder(
                dim=enc_n_embed,
                depth=enc_n_layer,
                heads=enc_n_head,
                layer_dropout=enc_dropout,
                rotary_pos_emb=True,
                ff_glu=True,
                ff_no_bias=True,
            ),
        )

        self.decoder = TransformerWrapper(
            num_tokens=tgt_vocab_size,
            max_seq_len=context_len,
            l2norm_embed=True,
            attn_layers=Decoder(
                dim=dec_n_embed,
                depth=dec_n_layer,
                heads=dec_n_head,
                layer_dropout=dec_dropout,
                cross_attend=True,
                rotary_pos_emb=True,
                ff_glu=True,
                ff_no_bias=True,
            ),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, src, tgt):
        # TODO: src_key_padding_mask needed for both encoder and decoder?
        src_key_padding_mask = src == self.pad_ix
        enc = self.encoder(src, mask=src_key_padding_mask, return_embeddings=True)

        attn_mask = get_causal_mask(tgt.size(1), device=tgt.device, boolean=True)
        out = self.decoder(
            tgt,
            attn_mask=attn_mask,
            context=enc,
            context_mask=src_key_padding_mask,
        )
        return out
