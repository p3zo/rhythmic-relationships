import torch
import torch.nn as nn


class TransformerEncoderDecoder(nn.Module):
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
    ):
        super().__init__()

        self.context_len = context_len

        self.src_token_embedding = nn.Embedding(
            num_embeddings=src_vocab_size,
            embedding_dim=n_embed,
        )
        self.tgt_token_embedding = nn.Embedding(
            num_embeddings=tgt_vocab_size,
            embedding_dim=n_embed,
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
        )
        self.output_layer = nn.Linear(n_embed, tgt_vocab_size)

        self.register_buffer("tril", torch.tril(torch.ones(context_len, context_len)))

    def forward(self, x, y):
        # B, T, C=N_embed
        enc_tok_emb = self.src_token_embedding(x)
        dec_tok_emb = self.tgt_token_embedding(y)

        # # T, C
        enc_pos_emb = self.position_embedding.weight[: x.shape[1]]
        dec_pos_emb = self.position_embedding.weight[: y.shape[1]]

        # B, T, C
        src = enc_tok_emb + enc_pos_emb
        tgt = dec_tok_emb + dec_pos_emb

        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1]).to(
            tgt.device
        )

        out = self.transformer(src, tgt, tgt_mask=tgt_mask)

        return self.output_layer(out)

    @torch.no_grad()
    def generate(self, x, y, max_new_tokens=32):
        # x is (B, T)

        self.eval()

        for _ in range(max_new_tokens):
            # Crop x to the last context_len tokens
            x_cond = x[:, -self.context_len :]
            y_cond = y[:, -self.context_len :]

            # Get the predictions
            logits = self(x_cond, y_cond)

            # Take the last predicted token
            logits = logits[:, -1, :]  # becomes (B, C)

            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)  # (B, C)

            # Sample from the distribution
            y_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append the sampled index to the running sequence
            y = torch.cat((y, y_next), dim=1)  # (B, T+1)

        self.train()

        return y
