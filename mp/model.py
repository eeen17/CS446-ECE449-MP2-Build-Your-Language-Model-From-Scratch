from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TransformerConfig:
    vocab_size: int
    max_seq_len: int
    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int
    dropout: float


def sinusoidal_position_encoding(
    seq_len: int,
    d_model: int,
) -> torch.Tensor:
    """Return sinusoidal positional encodings with shape [1, T, d_model].

    TODO:
    Implement the fixed sinusoidal positional encoding used in the original
    Transformer paper.

    Required behavior:
    1) Create position indices `0, 1, ..., seq_len - 1`.
    2) For every even hidden dimension, compute:
       `sin(position / 10000^(i / d_model))`
    3) For every odd hidden dimension, compute:
       `cos(position / 10000^(i / d_model))`
    4) Return the encodings with shape [1, seq_len, d_model].

    Important details:
    - return dtype should be `torch.float32`
    - you may assume `d_model` is even in this MP
    """
    raise NotImplementedError


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute masked multi-head self-attention.

        TODO:
        Implement multi-head causal self-attention for inputs of shape [B, T, d_model].

        Required behavior:
        1) Produce query, key, and value vectors for every token position.
        2) Split the model dimension into `n_heads` heads of size `head_dim`.
        3) Compute attention scores between every query position and every key position.
        4) Scale scores so they do not grow too large as `head_dim` increases.
        5) Apply a causal mask before normalization so position `t` can only attend to
           positions `<= t`.
        6) Normalize scores across the last dimension, apply attention dropout, and use the
           resulting weights to combine value vectors.
        7) Merge heads back to shape [B, T, d_model], apply the output projection, then apply
           residual dropout.

        Important details:
        - output shape must match the input shape
        - the mask must work for any sequence length up to `max_seq_len`
        - changing future tokens must not change logits for earlier positions
        """
        raise NotImplementedError


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pre-LN attention + MLP with residual connections.

        TODO:
        Implement the standard pre-layernorm Transformer block.

        Required behavior:
        1) Normalize the input before the attention sublayer.
        2) Add the attention output back to the original residual stream.
        3) Normalize the updated hidden states before the feed-forward sublayer.
        4) Add the feed-forward output back to the residual stream.

        Important details:
        - preserve shape [B, T, d_model]
        - both sublayers should be residual updates, not replacements
        """
        raise NotImplementedError


class TransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return logits with shape [B, T, vocab_size].

        TODO:
        Build the full decoder-only language model forward pass.

        Required behavior:
        1) Treat `input_ids` as shape [batch, seq_len].
        2) Reject sequences longer than `config.max_seq_len`.
        3) Build token embeddings and sinusoidal positional encodings for each time step.
        4) Add token and position information, then apply embedding dropout.
        5) Run the hidden states through every Transformer block in order.
        6) Apply the final layer norm and project to vocabulary logits.

        Important details:
        - output must have shape [B, T, vocab_size]
        - logits at each position correspond to the next-token prediction distribution
        """
        raise NotImplementedError

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively generate new tokens.

        TODO:
        Extend the prompt one token at a time.

        Required behavior:
        1) Repeatedly run the model on the current context and read the logits from the final
           time step only.
        2) If the current sequence is longer than `max_seq_len`, use only the most recent
           `max_seq_len` tokens as context.
        3) Choose the next token by:
           - greedy decoding when `temperature <= 0` or `top_k == 1`
           - otherwise sampling from the temperature-scaled distribution
        4) If `top_k` is provided, only allow sampling from the top-k logits.
        5) Append the chosen token and continue until `max_new_tokens` tokens are added.

        Important details:
        - return the full sequence including the original prompt
        - generation should work for batch size 1 and larger
        - greedy decoding and `top_k == 1` should behave identically
        """
        raise NotImplementedError
