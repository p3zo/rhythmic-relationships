import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, x_dim, h_dims, z_dim, conditional=False, y_dim=0):
        super().__init__()

        self.conditional = conditional

        # Encoder
        input_to_hidden = nn.Linear(x_dim + y_dim, h_dims[0])

        encoder_layers = [input_to_hidden, nn.ReLU()]
        for i, j in zip(h_dims, h_dims[1:]):
            encoder_layers.extend([nn.Linear(i, j), nn.ReLU()])

        self.encoder = nn.Sequential(*encoder_layers)

        self.hidden_to_mu = nn.Linear(h_dims[-1], z_dim)
        self.hidden_to_sigma = nn.Linear(h_dims[-1], z_dim)

        # Decoder
        z_to_hidden = nn.Linear(z_dim + y_dim, h_dims[-1])
        hidden_to_input = nn.Linear(h_dims[0], x_dim)

        decoder_layers = [z_to_hidden, nn.ReLU()]
        for i, j in zip(h_dims, h_dims[1:]):
            decoder_layers.extend([nn.Linear(j, i), nn.ReLU()])
        decoder_layers.append(hidden_to_input)

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x, c=None):
        if self.conditional:
            x = torch.cat((x, c), dim=-1)

        h = self.encoder(x)

        return self.hidden_to_mu(h), self.hidden_to_sigma(h)

    def decode(self, z, c=None, activation=None):
        if self.conditional:
            z = torch.cat((z, c), dim=-1)

        x = self.decoder(z)

        if activation == "sigmoid":
            return torch.sigmoid(x)

        return x

    def forward(self, x, c=None):
        mu, sigma = self.encode(x, c)

        epsilon = torch.randn_like(sigma)
        z_reparameterized = mu + sigma * epsilon
        x_reconstructed = self.decode(z_reparameterized, c)

        return x_reconstructed, mu, sigma


class RecurrentVAE(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim, context_len, y_dim, dropout):
        super().__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.context_len = context_len
        self.y_dim = y_dim

        # Encoder
        self.input_to_hidden = nn.RNN(
            input_size=context_len,
            hidden_size=h_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
            nonlinearity="relu",
        )

        # h_dim * 2 because of bidirectional
        self.hidden_to_mu = nn.Linear(h_dim * 2, z_dim)
        self.hidden_to_sigma = nn.Linear(h_dim * 2, z_dim)

        # Decoder
        self.z_to_h0 = nn.Sequential(nn.Linear(z_dim, h_dim), nn.Tanh())
        self.z_to_hidden = nn.RNN(
            input_size=z_dim,
            hidden_size=h_dim,
            batch_first=True,
            bidirectional=False,
            nonlinearity="relu",
        )
        # TODO: should hidden size of the second layer be context_len instead of x_dim? That would match encoder input size
        self.hidden_to_input = nn.RNN(
            input_size=h_dim,
            hidden_size=x_dim,
            batch_first=True,
            bidirectional=False,
            nonlinearity="relu",
        )
        # TODO: verify that -1 the correct dim for softmax
        self.softmax = nn.Softmax(dim=-1)

    def encode(self, x):
        unused_output, hidden = self.input_to_hidden(x)
        h = torch.cat((hidden[0], hidden[1]), dim=-1)
        return self.hidden_to_mu(h), self.hidden_to_sigma(h)

    def decode(self, z):
        # TODO: How to configure z_to_h0 given that z_to_hidden expects h0 to be of size (1, 1, 128)
        # h0 = self.z_to_h0(z)
        # output, h = self.z_to_hidden(z, h0)
        output, h = self.z_to_hidden(z)
        xoutput, xh = self.hidden_to_input(output)
        return self.softmax(xoutput)

    def forward(self, x):
        mu, sigma = self.encode(x)

        epsilon = torch.randn_like(sigma)
        z_reparameterized = mu + sigma * epsilon
        x_reconstructed = self.decode(z_reparameterized)

        return x_reconstructed, mu, sigma


class Head(nn.Module):
    """One head of self-attention
    This class was implemented following https://www.youtube.com/watch?v=kCc8FmEb1nY
    """

    def __init__(self, n_embed, head_size, context_len, dropout):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_len, context_len)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (Batch, Time-step, Channels)
        # output of size (Batch, Time-step, Head size)
        B, T, C = x.shape

        key = self.key(x)  # (B, T, H)
        query = self.query(x)  # (B, T, H)

        # Compute attention scores & scale (B, T, H) @ (B, H, T) -> (B, T, T)
        attention = torch.matmul(query, key.transpose(-2, -1)) * key.shape[-1] ** -0.5

        # Mask out attention for future tokens (B, T, T)
        attention = attention.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        attention = torch.softmax(attention, dim=-1)  # (B, T, T)
        attention = self.dropout(attention)

        # Perform the weighted aggregation of the values (B, T, H)
        value = self.value(x)
        out = torch.matmul(attention, value)

        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel
    This class was implemented following https://www.youtube.com/watch?v=kCc8FmEb1nY
    """

    def __init__(self, n_head, n_embed, head_size, context_len, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(
                    n_embed=n_embed,
                    head_size=head_size,
                    context_len=context_len,
                    dropout=dropout,
                )
                for _ in range(n_head)
            ]
        )
        self.proj = nn.Linear(head_size * n_head, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """This class was implemented following https://www.youtube.com/watch?v=kCc8FmEb1nY"""

    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4),
            nn.ReLU(),
            # projection layer going back into the residual pathway
            nn.Linear(n_embed * 4, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation
    This class was implemented following https://www.youtube.com/watch?v=kCc8FmEb1nY
    """

    def __init__(self, n_embed, context_len, n_head, dropout):
        super().__init__()
        head_size = n_embed // n_head
        self.sa_heads = MultiHeadAttention(
            n_head=n_head,
            n_embed=n_embed,
            head_size=head_size,
            context_len=context_len,
            dropout=dropout,
        )
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TransformerDecoder(nn.Module):
    """This class was implemented following https://www.youtube.com/watch?v=kCc8FmEb1nY"""

    def __init__(self, vocab_size, n_embed, context_len, n_head, n_layer, dropout):
        super().__init__()

        # TODO: Why do we need to add 1 to vocab_size? It's not necessary on `mps` but throws an error on `cpu` and `cuda`
        vocab_size = vocab_size + 1

        self.context_len = context_len

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(context_len, n_embed)
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embed=n_embed,
                    context_len=context_len,
                    n_head=n_head,
                    dropout=dropout,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_final = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        tok_emb = self.token_embedding_table(idx)  # B, T, C=N_embed

        pos_emb = self.position_embedding_table.weight[: idx.shape[1]]  # T, C

        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_final(x)  # (B, T, C)
        logits = self.lm_head(x)  # B, T, vocab_size

        return logits

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last context_len tokens
            idx_cond = idx[:, -self.context_len :]

            # Get the predictions
            logits = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)

            # apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)  # (B, C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


class TransformerDecoderNew(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_ff, dropout):
        super().__init__()

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        ln = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(
            decoder_layer=layer, num_layers=n_layer, norm=ln
        )

    def forward(self, x, y):
        return self.decoder(x, y)


class TransformerEncoder(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_ff, dropout):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        ln = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=layer, num_layers=n_layer, norm=ln
        )

    def forward(self, x):
        return self.encoder(x)


class TransformerEncoderDecoder(nn.Module):
    def __init__(
        self, context_len, vocab_size, n_embed, encoder_params, decoder_params
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, n_embed)

        self.encoder = TransformerEncoder(d_model=vocab_size, **encoder_params)
        self.decoder = TransformerDecoderNew(d_model=vocab_size, **decoder_params)

        self.pos_emb = nn.Embedding(context_len, n_embed)

    def forward(self, idx, idy):
        # B, T, C=N_embed
        enc_tok_emb = self.token_embedding(idx)
        dec_tok_emb = self.token_embedding(idy)

        # T, C
        enc_pos_emb = self.pos_emb.weight[: idx.shape[1]]
        dec_pos_emb = self.pos_emb.weight[: idy.shape[1]]

        # B, T, C
        x = enc_tok_emb + enc_pos_emb
        y = dec_tok_emb + dec_pos_emb

        x = self.encoder(x)

        # B, T, vocab_size
        logits = self.decoder(x, y)

        return logits


class TransformerEncoderDecoderNew(nn.Module):
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
        )

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

            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)

            # apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)  # (B, C)

            # sample from the distribution
            y_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # append sampled index to the running sequence
            y = torch.cat((y, y_next), dim=1)  # (B, T+1)

        self.train()

        return y
