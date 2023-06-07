import torch
import torch.nn as nn
from x_transformers import TransformerWrapper, Decoder, Encoder


class TransformerEncoderDecoder(nn.Module):
    def __init__(
        self,
        context_len,
        src_vocab_size,
        tgt_vocab_size,
        enc_n_embed,
        enc_n_layer,
        enc_n_head,
        enc_d_ff,
        enc_dropout,
        dec_n_embed,
        dec_n_layer,
        dec_n_head,
        dec_d_ff,
        dec_dropout,
        pad_ix,
    ):
        super().__init__()

        self.context_len = context_len
        self.pad_ix = pad_ix

        # TODO: try deepnorm=True
        # TODO: attn_sparse_topk=8 throws NameError
        # TODO: how to set ff dims?
        self.encoder = TransformerWrapper(
            num_tokens=src_vocab_size,
            max_seq_len=33,
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
            max_seq_len=33,
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
                attn_one_kv_head=True,  # only use one head for k/v, but multi-headed q
            ),
        )

    def forward(self, x, y):
        encoded = self.encoder(x, return_embeddings=True)
        return self.decoder(y, context=encoded)

    @torch.no_grad()
    def generate(
        self,
        seq_in,
        seq_out_start,
        seq_len,
        mask=None,
        attn_mask=None,
        **kwargs,
    ):
        self.eval()
        encodings = self.encoder(
            seq_in,
            mask=mask,
            attn_mask=attn_mask,
            return_embeddings=True,
        )
        out = self.decoder.generate(
            seq_out_start,
            seq_len,
            context=encodings,
            context_mask=mask,
            **kwargs,
        )
        self.train()
        return out


class TransformerEncoderDecoderOld(nn.Module):
    def __init__(
        self,
        context_len,
        src_vocab_size,
        tgt_vocab_size,
        n_embed,
        n_layer,
        n_head,
        d_ff,
        dropout,
        pad_ix,
    ):
        super().__init__()

        self.context_len = context_len
        self.pad_ix = pad_ix

        self.src_token_embedding = nn.Embedding(
            num_embeddings=src_vocab_size,
            embedding_dim=n_embed,
            padding_idx=pad_ix,
        )
        self.tgt_token_embedding = nn.Embedding(
            num_embeddings=tgt_vocab_size,
            embedding_dim=n_embed,
            padding_idx=pad_ix,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=context_len,
            embedding_dim=n_embed,
        )

        self.transformer = nn.Transformer(
            d_model=n_embed,
            nhead=n_head,
            num_encoder_layers=n_layer,
            num_decoder_layers=n_layer,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.output_layer_norm = nn.LayerNorm(n_embed)
        self.output_layer = nn.Linear(n_embed, tgt_vocab_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(context_len, context_len)))

    def forward(self, x, y):
        # B, T, C=N_embed
        enc_tok_emb = self.src_token_embedding(x)
        dec_tok_emb = self.tgt_token_embedding(y)

        # T, C
        enc_pos_emb = self.position_embedding.weight[: x.shape[1]]
        dec_pos_emb = self.position_embedding.weight[: y.shape[1]]

        # B, T, C
        src = enc_tok_emb + enc_pos_emb
        tgt = dec_tok_emb + dec_pos_emb

        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1]).to(
            tgt.device
        )

        out = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=y == self.pad_ix,
        )

        normed = self.output_layer_norm(out)

        return self.output_layer(normed)

    @torch.no_grad()
    def generate(self, x, y, max_new_tokens=32):
        # x is (B, T)

        self.eval()

        # Crop inputs to the last context_len tokens
        x_cond = x[:, -self.context_len :]

        for _ in range(max_new_tokens):
            y_cond = y[:, -self.context_len :]

            # Get the predictions
            logits = self(x_cond, y_cond)

            # Take the logits for the last tokens
            logits = logits[:, -1, :]  # becomes (B, C)

            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)  # (B, C)

            # Sample from the distribution
            y_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append the sampled index to the running sequence
            y = torch.cat((y, y_next), dim=1)  # (B, T+1)

        self.train()

        return y
