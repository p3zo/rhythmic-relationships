import torch
import torch.nn as nn
from x_transformers import TransformerWrapper, Decoder
from rhythmic_relationships.model_utils import get_causal_mask


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embed,
        context_len,
        n_head,
        n_layer,
        dropout,
        pad_ix,
    ):
        super().__init__()

        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embed = n_embed
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.context_len = context_len

        self.decoder = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=context_len,
            # l2norm_embed=True,
            attn_layers=Decoder(
                dim=n_embed,
                depth=n_layer,
                heads=n_head,
                layer_dropout=dropout,
                # rotary_pos_emb=True,
                # ff_glu=True,
                # ff_no_bias=True,
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

    def forward(self, x):
        attn_mask = get_causal_mask(x.size(1), device=x.device, boolean=True)
        out = self.decoder(x, attn_mask=attn_mask)
        return out

# class Head(nn.Module):
#     """One head of self-attention"""
#
#     def __init__(self, n_embed, head_size, context_len, dropout):
#         super().__init__()
#         self.key = nn.Linear(n_embed, head_size, bias=False)
#         self.query = nn.Linear(n_embed, head_size, bias=False)
#         self.value = nn.Linear(n_embed, head_size, bias=False)
#         self.register_buffer("tril", torch.tril(torch.ones(context_len, context_len)))
#
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         # input of size (Batch, Time-step, Channels)
#         # output of size (Batch, Time-step, Head size)
#         B, T, C = x.shape
#
#         key = self.key(x)  # (B, T, H)
#         query = self.query(x)  # (B, T, H)
#
#         # Compute attention scores & scale (B, T, H) @ (B, H, T) -> (B, T, T)
#         attention = torch.matmul(query, key.transpose(-2, -1)) * key.shape[-1] ** -0.5
#
#         # Mask out attention for future tokens (B, T, T)
#         attention = attention.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
#
#         attention = torch.softmax(attention, dim=-1)  # (B, T, T)
#         attention = self.dropout(attention)
#
#         # Perform the weighted aggregation of the values (B, T, H)
#         value = self.value(x)
#         out = torch.matmul(attention, value)
#
#         return out
#
#
# class MultiHeadAttention(nn.Module):
#     """Multiple heads of self-attention in parallel"""
#
#     def __init__(self, n_head, n_embed, head_size, context_len, dropout):
#         super().__init__()
#         self.heads = nn.ModuleList(
#             [
#                 Head(
#                     n_embed=n_embed,
#                     head_size=head_size,
#                     context_len=context_len,
#                     dropout=dropout,
#                 )
#                 for _ in range(n_head)
#             ]
#         )
#         self.proj = nn.Linear(head_size * n_head, n_embed)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         out = torch.cat([h(x) for h in self.heads], dim=-1)
#         out = self.dropout(self.proj(out))
#         return out
#
#
# class FeedForward(nn.Module):
#     def __init__(self, n_embed, dropout):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(n_embed, n_embed * 4),
#             nn.ReLU(),
#             # projection layer going back into the residual pathway
#             nn.Linear(n_embed * 4, n_embed),
#             nn.Dropout(dropout),
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
#
# class Block(nn.Module):
#     """Transformer block: communication followed by computation"""
#
#     def __init__(self, n_embed, context_len, n_head, dropout):
#         super().__init__()
#         head_size = n_embed // n_head
#         self.sa_heads = MultiHeadAttention(
#             n_head=n_head,
#             n_embed=n_embed,
#             head_size=head_size,
#             context_len=context_len,
#             dropout=dropout,
#         )
#         self.ffwd = FeedForward(n_embed, dropout)
#         self.ln1 = nn.LayerNorm(n_embed)
#         self.ln2 = nn.LayerNorm(n_embed)
#
#     def forward(self, x):
#         x = x + self.sa_heads(self.ln1(x))
#         x = x + self.ffwd(self.ln2(x))
#         return x
#
#
# class TransformerDecoderOld(nn.Module):
#     def __init__(
#         self, vocab_size, n_embed, context_len, n_head, n_layer, dropout, pad_ix
#     ):
#         super().__init__()
#
#         # TODO: Why do we add 1 to vocab_size? This is not necessary on `mps` but throws an error on `cpu` and `cuda`.
#         vocab_size = vocab_size + 1
#
#         self.context_len = context_len
#
#         self.token_embedding_table = nn.Embedding(
#             vocab_size, n_embed, padding_idx=pad_ix
#         )
#         self.position_embedding_table = nn.Embedding(
#             context_len, n_embed, padding_idx=pad_ix
#         )
#         self.blocks = nn.Sequential(
#             *[
#                 Block(
#                     n_embed=n_embed,
#                     context_len=context_len,
#                     n_head=n_head,
#                     dropout=dropout,
#                 )
#                 for _ in range(n_layer)
#             ]
#         )
#         self.ln_final = nn.LayerNorm(n_embed)
#         self.lm_head = nn.Linear(n_embed, vocab_size)
#         self.apply(self._init_weights)
#
#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.Embedding):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#
#     def forward(self, idx):
#         tok_emb = self.token_embedding_table(idx)  # B, T, C=N_embed
#
#         pos_emb = self.position_embedding_table.weight[: idx.shape[1]]  # T, C
#
#         x = tok_emb + pos_emb  # (B, T, C)
#         x = self.blocks(x)  # (B, T, C)
#         x = self.ln_final(x)  # (B, T, C)
#         logits = self.lm_head(x)  # B, T, vocab_size
#
#         return logits
