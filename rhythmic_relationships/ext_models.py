"""Adapted from https://github.com/YatingMusic/MuseMorphose"""

import torch
import torch.nn as nn


def get_causal_mask(seq_len, device):
    mask = (
        torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=device,
            )
        )
        == 1
    ).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    mask.requires_grad = False
    return mask


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
        if hasattr(m, "weight"):
            weight_init_normal(m.weight, 0.01)
    elif classname.find("LayerNorm") != -1:
        if hasattr(m, "weight"):
            nn.init.normal_(m.weight, 1.0, 0.01)
        if hasattr(m, "bias") and m.bias is not None:
            bias_init(m.bias)
    elif classname.find("GRU") != -1:
        for param in m.parameters():
            if len(param.shape) >= 2:
                weight_init_orthogonal(param, 0.01)
            else:
                bias_init(param)


class TokenEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, pad_ix):
        super().__init__()

        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj
        self.emb_scale = d_proj**0.5

        self.emb_lookup = nn.Embedding(
            num_embeddings=n_token,
            embedding_dim=d_embed,
            padding_idx=pad_ix,
        )

        if d_proj != d_embed:
            self.emb_proj = nn.Linear(d_embed, d_proj, bias=False)
        else:
            self.emb_proj = None

    def forward(self, inp_tokens):
        inp_emb = self.emb_lookup(inp_tokens)

        if self.emb_proj is not None:
            inp_emb = self.emb_proj(inp_emb)

        return inp_emb.mul_(self.emb_scale)


class VAETransformerEncoder(nn.Module):
    def __init__(
        self,
        n_layer,
        n_head,
        d_model,
        d_ff,
        d_vae_latent,
        dropout,
    ):
        """Adapted from https://github.com/YatingMusic/MuseMorphose/blob/main/model/transformer_encoder.py"""
        super().__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_vae_latent = d_vae_latent
        self.dropout = dropout

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            n_head,
            d_ff,
            dropout,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, n_layer)

        self.fc_mu = nn.Linear(d_model, d_vae_latent)
        self.fc_logvar = nn.Linear(d_model, d_vae_latent)

    def forward(self, x):
        out = self.encoder(x)
        hidden_out = out[0, :, :]
        mu, logvar = self.fc_mu(hidden_out), self.fc_logvar(hidden_out)
        return hidden_out, mu, logvar


class VAETransformerDecoder(nn.Module):
    def __init__(
        self,
        n_layer,
        n_head,
        d_model,
        d_ff,
        d_seg_emb,
        pad_ix,
        dropout,
    ):
        """Adapted from https://github.com/YatingMusic/MuseMorphose/blob/main/model/musemorphose.py"""
        super().__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_seg_emb = d_seg_emb
        self.pad_ix = pad_ix
        self.dropout = dropout
        self.seg_emb_proj = nn.Linear(d_seg_emb, d_model, bias=False)
        self.decoder_layers = nn.ModuleList()
        for i in range(n_layer):
            self.decoder_layers.append(
                nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout)
            )

    def forward(self, x, seg_emb):
        attn_mask = get_causal_mask(x.size(0), device=x.device)

        seg_emb = self.seg_emb_proj(seg_emb)

        out = x
        for i in range(self.n_layer):
            out += seg_emb
            # TODO: pass key padding mask?
            out = self.decoder_layers[i](x, src_mask=attn_mask)

        return out


class MuseMorphoseAdapted(nn.Module):
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
        d_embed,
        src_vocab_size,
        tgt_vocab_size,
        pad_ix,
        context_len,
        enc_dropout,
        dec_dropout,
        emb_dropout,
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
        self.src_vocab_size = src_vocab_size

        self.pad_ix = pad_ix

        self.src_token_embedding = TokenEmbedding(
            n_token=src_vocab_size,
            d_embed=d_embed,
            d_proj=enc_d_model,
            pad_ix=pad_ix,
        )
        self.tgt_token_embedding = TokenEmbedding(
            n_token=tgt_vocab_size,
            d_embed=d_embed,
            d_proj=dec_d_model,
            pad_ix=pad_ix,
        )
        self.d_embed = d_embed
        self.pe = nn.Embedding(
            num_embeddings=context_len,
            embedding_dim=d_embed,
        )
        self.dec_out_proj = nn.Linear(dec_d_model, tgt_vocab_size)
        self.encoder = VAETransformerEncoder(
            n_layer=enc_n_layer,
            n_head=enc_n_head,
            d_model=enc_d_model,
            d_ff=enc_d_ff,
            d_vae_latent=d_vae_latent,
            dropout=enc_dropout,
        )

        self.decoder = VAETransformerDecoder(
            n_layer=dec_n_layer,
            n_head=dec_n_head,
            d_model=dec_d_model,
            d_ff=dec_d_ff,
            d_seg_emb=d_vae_latent,
            pad_ix=pad_ix,
            dropout=dec_dropout,
        )

        self.emb_dropout = nn.Dropout(emb_dropout)
        self.apply(weights_init)

    def reparameterize(self, mu, logvar, use_sampling=True, sampling_var=1.0):
        std = torch.exp(0.5 * logvar).to(mu.device)
        if use_sampling:
            eps = torch.randn_like(std, device=mu.device) * sampling_var
        else:
            eps = torch.zeros_like(std, device=mu.device)
        return eps * std + mu

    def get_sampled_latent(self, inp, use_sampling, sampling_var):
        enc_tok_emb = self.src_token_embedding(inp)
        enc_pos_emb = self.pe.weight[: inp.shape[1]]
        enc_inp = self.emb_dropout(enc_tok_emb) + enc_pos_emb

        _, mu, logvar = self.encoder(enc_inp)
        mu, logvar = mu.reshape(-1, mu.size(-1)), logvar.reshape(-1, mu.size(-1))

        vae_latent = self.reparameterize(
            mu,
            logvar,
            use_sampling=use_sampling,
            sampling_var=sampling_var,
        )

        return vae_latent

    @torch.no_grad()
    def generate(self, dec_seg_emb, ctx):
        dec_tok_emb = self.tgt_token_embedding(ctx)
        dec_pos_emb = self.pe.weight[: ctx.shape[1]]
        dec_inp = self.emb_dropout(dec_tok_emb) + dec_pos_emb

        out = self.decoder(dec_inp, dec_seg_emb)
        out = self.dec_out_proj(out)

        return out

    def forward(self, x, y):
        enc_tok_emb = self.src_token_embedding(x)
        dec_tok_emb = self.tgt_token_embedding(y)

        enc_pos_emb = self.pe.weight[: x.shape[1]]
        dec_pos_emb = self.pe.weight[: y.shape[1]]

        enc_inp = enc_tok_emb + enc_pos_emb
        dec_inp = dec_tok_emb + dec_pos_emb

        _, mu, logvar = self.encoder(enc_inp)

        vae_latent = self.reparameterize(mu, logvar)

        dec_out = self.decoder(dec_inp, vae_latent)
        dec_logits = self.dec_out_proj(dec_out)

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
