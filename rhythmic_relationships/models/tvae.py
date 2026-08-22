"""Adapted from https://github.com/YatingMusic/MuseMorphose"""

import torch
import torch.nn as nn
from x_transformers import TransformerWrapper, Decoder, Encoder


def get_causal_mask(sz, device, boolean=False):
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask.requires_grad = False
    if boolean:
        return mask
    return (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )


def weight_init_normal(weight, normal_std):
    nn.init.normal_(weight, 0.0, normal_std)


def weight_init_orthogonal(weight, gain):
    nn.init.orthogonal_(weight, gain)


def bias_init(bias):
    nn.init.constant_(bias, 0.0)


def weights_init(m):
    """From https://github.com/YatingMusic/MuseMorphose/blob/main/model/transformer_helpers.py"""
    classname = m.__class__.__name__

    if classname.find("Linear") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            weight_init_normal(m.weight, 0.01)
        if hasattr(m, "bias") and m.bias is not None:
            bias_init(m.bias)
    elif classname.find("Embedding") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            weight_init_normal(m.weight, 0.01)
    elif classname.find("LayerNorm") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight, 1.0, 0.01)
        if hasattr(m, "bias") and m.bias is not None:
            bias_init(m.bias)
    elif classname.find("GRU") != -1:
        for param in m.parameters():
            if len(param.shape) >= 2:
                weight_init_orthogonal(param, 0.01)
            else:
                bias_init(param)


class VAETransformerEncoder(nn.Module):
    def __init__(
        self,
        n_layer,
        n_head,
        d_model,
        d_ff,
        d_vae_latent,
        dropout,
        vocab_size,
        context_len,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_vae_latent = d_vae_latent
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.context_len = context_len

        self.encoder = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=context_len,
            l2norm_embed=True,
            attn_layers=Encoder(
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

        self.fc_mu = nn.Linear(d_model, d_vae_latent)
        self.fc_logvar = nn.Linear(d_model, d_vae_latent)

    def forward(self, x):
        h = self.encoder(x, return_embeddings=True)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return h, mu, logvar


class VAETransformerDecoder(nn.Module):
    def __init__(
        self,
        n_layer,
        n_head,
        d_model,
        d_ff,
        d_latent,
        pad_ix,
        dropout,
        vocab_size,
        context_len,
    ):
        """Adapted from https://github.com/YatingMusic/MuseMorphose/blob/main/model/musemorphose.py"""
        super().__init__()

        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_latent = d_latent
        self.pad_ix = pad_ix
        self.dropout = dropout
        self.latent_proj = nn.Linear(d_latent, d_model, bias=False)
        self.vocab_size = vocab_size
        self.context_len = context_len

        self.decoder = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=context_len,
            l2norm_embed=True,
            attn_layers=Decoder(
                dim=d_model,
                depth=n_layer,
                heads=n_head,
                layer_dropout=dropout,
                cross_attend=True,
                rotary_pos_emb=True,
                ff_glu=True,
                ff_no_bias=True,
                # x-transformers sizes the feed-forward as a multiple of the model dim
                ff_mult=d_ff / d_model,
            ),
        )

    def forward(self, x, latent):
        latent_proj = self.latent_proj(latent)
        attn_mask = get_causal_mask(x.size(1), device=x.device, boolean=True)
        out = self.decoder(x, context=latent_proj, attn_mask=attn_mask)
        return out


class VAETransformer(nn.Module):
    def __init__(
        self,
        enc_n_layer,
        enc_n_head,
        enc_d_model,
        enc_d_ff,
        dec_n_layer,
        dec_n_head,
        dec_d_model,
        dec_d_ff,
        d_vae_latent,
        src_vocab_size,
        tgt_vocab_size,
        pad_ix,
        context_len,
        enc_dropout,
        dec_dropout,
    ):
        """Adapted from https://github.com/YatingMusic/MuseMorphose/blob/main/model/musemorphose.py"""
        super().__init__()
        self.enc_n_layer = enc_n_layer
        self.enc_n_head = enc_n_head
        self.enc_d_model = enc_d_model
        self.enc_d_ff = enc_d_ff
        self.enc_dropout = enc_dropout

        self.dec_n_layer = dec_n_layer
        self.dec_n_head = dec_n_head
        self.dec_d_model = dec_d_model
        self.dec_d_ff = dec_d_ff
        self.dec_dropout = dec_dropout

        self.d_vae_latent = d_vae_latent
        self.pad_ix = pad_ix

        self.encoder = VAETransformerEncoder(
            n_layer=enc_n_layer,
            n_head=enc_n_head,
            d_model=enc_d_model,
            d_ff=enc_d_ff,
            d_vae_latent=d_vae_latent,
            dropout=enc_dropout,
            vocab_size=src_vocab_size,
            context_len=context_len,
        )

        self.decoder = VAETransformerDecoder(
            n_layer=dec_n_layer,
            n_head=dec_n_head,
            d_model=dec_d_model,
            d_ff=dec_d_ff,
            d_latent=d_vae_latent,
            pad_ix=pad_ix,
            dropout=dec_dropout,
            vocab_size=tgt_vocab_size,
            context_len=context_len,
        )

        for layer in (self.encoder.fc_mu, self.encoder.fc_logvar, self.decoder.latent_proj):
            weights_init(layer)

    def reparameterize(self, mu, logvar, use_sampling=True, sampling_var=1.0):
        std = torch.exp(0.5 * logvar).to(mu.device)
        if use_sampling:
            eps = torch.randn_like(std, device=mu.device) * sampling_var
        else:
            eps = torch.zeros_like(std, device=mu.device)
        return eps * std + mu

    def get_sampled_latent(self, x, use_sampling, sampling_var):
        """Returns a (batch, seq_len, d_vae_latent) latent sequence for a batch of inputs"""
        _, mu, logvar = self.encoder(x)

        latent = self.reparameterize(
            mu,
            logvar,
            use_sampling=use_sampling,
            sampling_var=sampling_var,
        )

        return latent

    @torch.no_grad()
    def generate(self, y, latent):
        out = self.decoder(y, latent)
        return out

    def forward(self, x, y):
        _, mu, logvar = self.encoder(x)
        latent = self.reparameterize(mu, logvar)

        dec_logits = self.decoder(y, latent)

        return mu, logvar, dec_logits

    def compute_loss(self, mu, logvar, beta, fb_lambda, dec_logits, dec_tgt):
        recons_loss = nn.functional.cross_entropy(
            dec_logits.view(-1, dec_logits.size(-1)),
            dec_tgt.contiguous().view(-1),
            ignore_index=self.pad_ix,
            reduction="mean",
        ).float()

        kl_raw = -0.5 * (1 + logvar - mu**2 - logvar.exp()).mean(dim=0)
        kl_before_free_bits = kl_raw.mean()
        kl_after_free_bits = kl_raw.clamp(min=fb_lambda)
        kldiv_loss = kl_after_free_bits.mean()

        return {
            "beta": beta,
            "total_loss": recons_loss + beta * kldiv_loss,
            "kldiv_loss": kldiv_loss,
            "kldiv_raw": kl_before_free_bits,
            "recons_loss": recons_loss,
        }
