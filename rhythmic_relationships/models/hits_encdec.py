
"""Adapted from https://github.com/YatingMusic/MuseMorphose"""

import torch
import torch.nn as nn
from x_transformers import TransformerWrapper, Decoder, Encoder





def weight_init_normal(weight, normal_std):
    nn.init.normal_(weight, 0.0, normal_std)


def weight_init_orthogonal(weight, gain):
    nn.init.orthogonal_(weight, gain)


def bias_init(bias):
    nn.init.constant_(bias, 0.0)


"""From https://github.com/YatingMusic/MuseMorphose/blob/main/model/transformer_helpers.py"""


def weights_init(m):
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
    def __init__(self, n_token, d_embed, d_proj):
        super().__init__()

        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj
        self.emb_scale = d_proj**0.5

        self.emb_lookup = nn.Embedding(
            num_embeddings=n_token,
            embedding_dim=d_embed,
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

        # TODO: how to set ff dims?
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
        dropout,
        vocab_size,
        context_len,
    ):
        super().__init__()

        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_latent = d_latent
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
            ),
        )

    def forward(self, x, latent):
        latent_proj = self.latent_proj(latent)
        attn_mask = get_causal_mask(x.size(1), device=x.device, boolean=True)
        out = self.decoder(x, context=latent_proj, attn_mask=attn_mask)
        return out


class TVAE(nn.Module):
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
        context_len,
        enc_dropout,
        dec_dropout,
        emb_dropout,
    ):
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

        self.src_token_embedding = TokenEmbedding(
            n_token=src_vocab_size,
            d_embed=d_embed,
            d_proj=enc_d_model,
        )
        self.tgt_token_embedding = TokenEmbedding(
            n_token=tgt_vocab_size,
            d_embed=d_embed,
            d_proj=dec_d_model,
        )
        self.d_embed = d_embed
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.pe = nn.Embedding(
            num_embeddings=context_len,
            embedding_dim=d_embed,
        )

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
            dropout=dec_dropout,
            vocab_size=tgt_vocab_size,
            context_len=context_len,
        )

        self.apply(weights_init)

    def reparameterize(self, mu, logvar, use_sampling=True, sampling_var=1.0):
        std = torch.exp(0.5 * logvar).to(mu.device)
        if use_sampling:
            eps = torch.randn_like(std, device=mu.device) * sampling_var
        else:
            eps = torch.zeros_like(std, device=mu.device)
        return eps * std + mu

    def get_sampled_latent(self, x, use_sampling, sampling_var):
        _, mu, logvar = self.encoder(x)

        # TODO: is this reshape necessary?
        # mu, logvar = mu.reshape(-1, mu.size(-1)), logvar.reshape(-1, mu.size(-1))

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
