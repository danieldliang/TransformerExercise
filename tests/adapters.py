from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

import torch
import math
from einops import rearrange, einsum

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        std = math.sqrt(2 / (in_features + out_features))
        self.weights = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.zeros(out_features, in_features), mean=0, std=std, a = -3 * std, b = 3 * std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")
    
class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.params = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.zeros(num_embeddings, embedding_dim), mean=0, std=1, a = -3, b = 3))
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.params[token_ids]

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        if dtype is not None and device is not None:
            self.params = torch.nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))
        elif dtype is not None:
            self.params = torch.nn.Parameter(torch.ones(d_model, dtype=dtype))
        elif device is not None:
            self.params = torch.nn.Parameter(torch.ones(d_model, device=device))
        else:
            self.params = torch.nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.rsqrt(torch.sum(torch.square(x), dim = -1, keepdim=True) / self.d_model + self.eps)
        result = (x * rms) * self.params.to(torch.float32)
        return result.to(in_dtype)

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    layer = Linear(d_in, d_out)
    layer.load_state_dict({"weights": weights})
    return layer(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    embed = Embedding(vocab_size, d_model)
    embed.load_state_dict({"params": weights})
    return embed(token_ids)

class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = int(64 * (((8 * d_model) / 3 + 63) // 64))
        self.w1_weight = torch.ones(self.d_ff, self.d_model)
        self.w2_weight = torch.ones(self.d_model, self.d_ff)
        self.w3_weight = torch.ones(self.d_ff, self.d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = einsum(x, self.w1_weight, "... d_model, d_ff d_model -> ... d_ff")
        w3_x = einsum(x, self.w3_weight, "... d_model, d_ff d_model -> ... d_ff")

        return einsum(self.w2_weight, w1_x * torch.sigmoid(w1_x) * w3_x, "d_model d_ff, ... d_ff -> ... d_model")

def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    swiglu = PositionWiseFeedForward(d_model)
    swiglu.w1_weight = w1_weight
    swiglu.w2_weight = w2_weight
    swiglu.w3_weight = w3_weight

    swiglu.d_ff = d_ff

    return swiglu(in_features)

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    QKT = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    QKT /= torch.sqrt(torch.tensor(Q.shape[-1]))
    if mask is not None:
        masked_QKT = torch.where(mask, QKT, -torch.inf)
    else:
        masked_QKT = QKT
    attention = einsum(SoftMax(masked_QKT, dim=-1), V, "... queries keys, ... keys d_v -> ... queries d_v")
    return attention

def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    return scaled_dot_product_attention(Q, K, V, mask)

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int=2048, theta: float = 1009):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.d_k = self.d_v = d_model // num_heads

        self.W_QKV = torch.nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_O = torch.nn.Linear(d_model, d_model, bias=False)

        self.rope = RoPE(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)

    def forward(self, 
                x: Float[torch.Tensor, " ... seq_len d_model"], 
                token_positions: Int[torch.Tensor, " ... seq_len"] | None = None
                ) -> Float[Tensor, " ... seq_len d_model"]:
        seq_len = x.shape[-2]
        QKV = self.W_QKV(x)
        QKV = rearrange(QKV, " ... seq_len (qkv num_heads d_k) -> ... seq_len qkv num_heads d_k", qkv=3, num_heads=self.num_heads, d_k=self.d_k)
        Q = rearrange(QKV[..., 0,:,:], " ... seq_len num_heads d_k -> ... num_heads seq_len d_k")
        K = rearrange(QKV[..., 1,:,:], " ... seq_len num_heads d_k -> ... num_heads seq_len d_k")
        V = rearrange(QKV[..., 2,:,:], " ... seq_len num_heads d_k -> ... num_heads seq_len d_k")

        if token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        mask = torch.empty((seq_len, seq_len), dtype=bool)
        for i in range(seq_len):
            for j in range(0, i + 1):
                mask[i][j] = True
            for j in range(i + 1, seq_len):
                mask[i][j] = False
        
        multihead = scaled_dot_product_attention(Q, K, V, mask)
        multihead = rearrange(multihead, "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)")
        return self.W_O(multihead)

def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    mhsa = MultiHeadSelfAttention(d_model, num_heads, in_features.shape[-2])
    qkv = torch.stack([q_proj_weight, k_proj_weight, v_proj_weight])
    qkv = rearrange(qkv, "three d_k d_in -> (three d_k) d_in", three = 3)
    mhsa.W_QKV.weight.data.copy_(qkv)
    mhsa.W_O.weight.data.copy_(o_proj_weight)

    return mhsa(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    mhsa = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta)
    qkv = torch.stack([q_proj_weight, k_proj_weight, v_proj_weight])
    qkv = rearrange(qkv, "three d_k d_in -> (three d_k) d_in", three = 3)
    mhsa.W_QKV.weight.data.copy_(qkv)
    mhsa.W_O.weight.data.copy_(o_proj_weight)

    return mhsa(in_features, token_positions)


class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        angles = torch.empty((max_seq_len, d_k // 2), dtype=torch.float64)
        for i in range(max_seq_len):
            for k in range(d_k // 2):
                angles[i][k] = i / (theta ** ((2 * k) / d_k))
        
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        original = x.dtype
        x = x.to(torch.float64)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out.to(dtype=original)

def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    
    rope = RoPE(theta, d_k, max_seq_len)

    return rope(in_query_or_key, token_positions)

class TransformerBlock(torch.nn.Module):
    def __init__(self,
                d_model: int,
                num_heads: int,
                d_ff: int,
                max_seq_len: int,
                theta: float,
                weights: dict[str, Tensor],
                ) -> Float[Tensor, " batch sequence_length d_model"]:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.weights = weights

        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.mhsa = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
        )
        self.pwff = PositionWiseFeedForward(d_model)
        self.pwff.d_ff = d_ff

        self.ln1.load_state_dict({"params": weights["ln1.weight"]})
        self.ln2.load_state_dict({"params": weights["ln2.weight"]})

        qkv = torch.cat([weights["attn.q_proj.weight"], weights["attn.k_proj.weight"], weights["attn.v_proj.weight"]], dim=0)
        self.mhsa.W_QKV.weight.data.copy_(qkv)
        self.mhsa.W_O.weight.data.copy_(weights["attn.output_proj.weight"])

        self.pwff.w1_weight = weights["ffn.w1.weight"]
        self.pwff.w2_weight = weights["ffn.w2.weight"]
        self.pwff.w3_weight = weights["ffn.w3.weight"]
    
    def forward(self, in_features: Float[Tensor, " batch sequence_length d_model"]
                ) -> Float[Tensor, " batch sequence_length d_model"]:
        seq_len = in_features.shape[-2]
        token_positions = torch.arange(seq_len, device=in_features.device).unsqueeze(0)
        token_positions = token_positions.expand(in_features.shape[0], seq_len)

        out1 = in_features + self.mhsa(self.ln1(in_features), token_positions)
        out2 = out1 + self.pwff(self.ln2(out1))
        return out2
    
class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weights: dict[str, Tensor],
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.weights = weights

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.token_embeddings.load_state_dict({"params": weights["token_embeddings.weight"]})

        self.layers = torch.nn.ModuleList()
        for layer_idx in range(num_layers):
            prefix = f"layers.{layer_idx}."
            layer_weights = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
            tb = TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    weights=layer_weights,
                )
            self.layers.append(tb)

        self.ln_final = RMSNorm(d_model)
        self.ln_final.load_state_dict({"params": weights["ln_final.weight"]})

        self.lm_head = Linear(d_model, vocab_size)
        self.lm_head.load_state_dict({"weights": weights["lm_head.weight"]})

    def forward(
        self, in_indices: Int[Tensor, " batch_size sequence_length"]
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        x = self.token_embeddings(in_indices)
        for block in self.layers:
            x = block(x)
        x = self.ln_final(x)
        return self.lm_head(x)

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    ln1 = RMSNorm(d_model)
    ln1.load_state_dict({"params": weights["ln1.weight"]})
    ln2 = RMSNorm(d_model)
    ln2.load_state_dict({"params": weights["ln2.weight"]})

    mhsa = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta)
    qkv = torch.stack([weights["attn.q_proj.weight"], weights["attn.k_proj.weight"], weights["attn.v_proj.weight"]])
    qkv = rearrange(qkv, "three d_k d_in -> (three d_k) d_in", three=3)
    mhsa.W_QKV.weight.data.copy_(qkv)
    mhsa.W_O.weight.data.copy_(weights["attn.output_proj.weight"])

    pwff = PositionWiseFeedForward(d_model)
    pwff.d_ff = d_ff
    pwff.w1_weight = weights["ffn.w1.weight"]
    pwff.w2_weight = weights["ffn.w2.weight"]
    pwff.w3_weight = weights["ffn.w3.weight"]

    seq_len = in_features.shape[-2]
    token_positions = torch.arange(seq_len, device=in_features.device).unsqueeze(0)
    token_positions = token_positions.expand(in_features.shape[0], seq_len)

    x = in_features + mhsa(ln1(in_features), token_positions)
    x = x + pwff(ln2(x))
    return x

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    embed = Embedding(vocab_size, d_model)
    embed.load_state_dict({"params": weights["token_embeddings.weight"]})
    x = embed(in_indices)

    seq_len = x.shape[-2]
    token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
    token_positions = token_positions.expand(x.shape[0], seq_len)

    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}."

        ln1 = RMSNorm(d_model)
        ln1.load_state_dict({"params": weights[f"{prefix}ln1.weight"]})
        ln2 = RMSNorm(d_model)
        ln2.load_state_dict({"params": weights[f"{prefix}ln2.weight"]})

        mhsa = MultiHeadSelfAttention(d_model, num_heads, max_seq_len=context_length, theta=rope_theta)
        qkv = torch.stack([weights[f"{prefix}attn.q_proj.weight"], weights[f"{prefix}attn.k_proj.weight"], weights[f"{prefix}attn.v_proj.weight"]])
        qkv = rearrange(qkv, "three d_k d_in -> (three d_k) d_in", three=3)
        mhsa.W_QKV.weight.data.copy_(qkv)
        mhsa.W_O.weight.data.copy_(weights[f"{prefix}attn.output_proj.weight"])

        pwff = PositionWiseFeedForward(d_model)
        pwff.d_ff = d_ff
        pwff.w1_weight = weights[f"{prefix}ffn.w1.weight"]
        pwff.w2_weight = weights[f"{prefix}ffn.w2.weight"]
        pwff.w3_weight = weights[f"{prefix}ffn.w3.weight"]

        x = x + mhsa(ln1(x), token_positions)
        x = x + pwff(ln2(x))

    ln_final = RMSNorm(d_model)
    ln_final.load_state_dict({"params": weights["ln_final.weight"]})
    x = ln_final(x)

    lm_head = Linear(d_model, vocab_size)
    lm_head.load_state_dict({"weights": weights["lm_head.weight"]})
    return lm_head(x)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rmsnorm = RMSNorm(d_model, eps)
    rmsnorm.load_state_dict({"params": weights})
    return rmsnorm(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError

def SoftMax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    # Edge case: if negative infinity mask is applied to the entire row, return all 0's instead of NaNs
    x = torch.where(in_features != -torch.inf, in_features - torch.amax(in_features, dim=dim, keepdim=True), -torch.inf)
    # In case negative infinity mask is applied to the entire row, we need to check that the sum is strictly greater than 0 
    return torch.where(torch.sum(torch.exp(x), dim=dim, keepdim=True) > 0, torch.exp(x) / torch.sum(torch.exp(x), dim=dim, keepdim=True), 0)

def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    return SoftMax(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError

from collections.abc import Iterable, Iterator
import regex as re
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class Tokenizer:
    def __init__(self, 
                vocab: dict[int, bytes],
                merges: list[tuple[bytes, bytes]],
                special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.invert_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        if special_tokens:
            self.special_tokens = special_tokens
        else:
            self.special_tokens = []
        
        self.merge_order = {m: i for i, m in enumerate(self.merges)}
        self.special_order = sorted(self.special_tokens, key=len, reverse=True)
        self.special_hash = set(self.special_order)
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        import json

        with open(vocab_filepath, encoding="utf-8") as fin:
            vocab = json.load(fin)
        
        with open(merges_filepath, encoding="utf-8") as fin:
            merges = json.load(fin)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    def encode(self, text: str) -> list[int]:
        return list(self.encode_iterable([text]))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for cur in iterable:
            text = cur.replace("\r\n", "\n").replace("\r", "\n")
            if self.special_order:
                # Remove all the special tokens before pretokenization
                splitter = re.compile("(" + "|".join(re.escape(token) for token in self.special_order) + ")")
                split_text = splitter.split(text)
            else:
                split_text = [text]

            for segment in split_text:
                if segment in self.special_hash:
                    yield self.invert_vocab[segment.encode("utf-8")]
                    continue
                for i in PAT.finditer(segment):
                    pretoken = i.group(0)

                    btext = [bytes([b]) for b in pretoken.encode("utf-8")]
                    # Either set up priority queue or quadratic solution

                    while True:
                        min_count = len(self.merge_order)
                        merge = None
                        for j in range(len(btext) - 1):
                            cur_merge = (btext[j], btext[j + 1])
                            cur_count = self.merge_order.get(cur_merge, min_count)
                            if min_count > cur_count:
                                min_count = cur_count
                                merge = cur_merge
                        if min_count == len(self.merge_order):
                            break

                        btemp = []
                        j = 0
                        while j < len(btext):
                            if (j < len(btext) - 1) and ((btext[j], btext[j + 1]) == merge):
                                btemp.append(btext[j] + btext[j + 1])
                                j += 2
                            else:
                                btemp.append(btext[j])
                                j += 1
                        btext = btemp
                
                    # newbtext = []
                    # for merge in self.merges:
                    #     for i in range(len(btext) - 1):
                    #         if merge == (btext[i], btext[i + 1]):
                    #             flag = True
                    #             newbtext.append(btext[i] + btext[i + 1])
                    #             i += 1
                    #         else:
                    #             flag = False
                    #             newbtext.append(btext[i])
                    #     if not flag:
                    #         newbtext.append(btext[-1])
                    #     btext = newbtext
                    #     newbtext = []
                    
                    for j in btext:
                        yield self.invert_vocab[j]
        
    def decode(self, ids: list[int]) -> str:
        cur = bytearray()
        for id in ids:
            cur.extend(self.vocab[id])
        
        return cur.decode("utf-8", errors="replace")

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)

# from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed
import regex as re
import os
from typing import BinaryIO
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def find_chunk_boundaries(
    file: str,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file = open(file, "rb")
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    file.close()
    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def _process_chunk(file, start, end, special_tokens):
    # Consider streaming the file inputs (However, this makes special token splitting particularly complicated)
    # Must split special tokens BEFORE regex nextiter streaming, 
    # since that would cause splitting conflicts otherwise.
    # Fixed-mini-chunking as done in find_chunk_boundaries risks splitting at exactly special tokens
    # Thus, either adding alpha tunable parameter or repeated calls to find_chunk_boundaries are viable potential solutions.
    f = open(file, "rb")
    f.seek(start)
    chunk = f.read(end - start).decode("utf-8", errors="replace")
    chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
    # Remove all the special tokens before pretokenization
    splitter = re.compile("|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)))
    split_chunk = splitter.split(chunk)

    pretokens = {}
    for segment in split_chunk:
        for i in PAT.finditer(segment):
            pretoken = i.group(0)
            pretokens[pretoken] = pretokens.get(pretoken, 0) + 1
    f.close()
    return pretokens

def TrainBPE(input_path, vocab_size, special_tokens):
    num_processes = 4

    loop_limit = 10000
    memory_limit = 5000000 # technically approximate memory average

    # Arbitrary tunable parameter (make this fixed or tunable based on memory limit)
    alpha = min(loop_limit, (os.path.getsize(input_path) + memory_limit - 1) // memory_limit)

    # Arbitrarily set the zero-th index of special_tokens to be the end-of-text token (assume special_tokens is nonempty)
    boundaries =  find_chunk_boundaries(input_path, alpha, special_tokens[0].encode("utf-8"))
    
    jobs = []
    chunk_freqs = {}
    # with ProcessPoolExecutor(max_workers=num_processes) as ex:
    with ThreadPoolExecutor(max_workers=num_processes) as ex:
        for i in range(len(boundaries) - 1):
            jobs.append(ex.submit(_process_chunk, input_path, boundaries[i], boundaries[i + 1], special_tokens))

        for j in as_completed(jobs):
            print("test: ", j)
            for k, v in j.result().items():
                chunk_freqs[k] = chunk_freqs.get(k, 0) + v
    
    freqs = {}
    for token, cnt in chunk_freqs.items():
        byte_token = token.encode("utf-8")
        # print("Token:", token)
        # print("Byte token:", byte_token)
        bchars = tuple(bytes([b]) for b in byte_token)
        freqs[bchars] = cnt
    
    # Now that pretoken frequencies are acculumated and tupled,
    # We can start merging most frequent bytes
    vocab = {i : bytes([i]) for i in range(256)}
    cnt = 256
    for token in special_tokens:
        binary_token = token.encode("utf-8")
        if binary_token not in vocab.values():
            vocab[cnt] = binary_token
            cnt += 1

    pair_freqs = {}
    pair_index = {}
    for bchars, freq in freqs.items():
        for i in range(len(bchars) - 1):
            p = (bchars[i], bchars[i + 1])
            pair_freqs[p] = pair_freqs.get(p, 0) + freq
            if p not in pair_index:
                pair_index[p] = set()
            pair_index[p].add(bchars)

    merges = []
    while len(vocab) < vocab_size and pair_freqs:
        byte_pair = max(pair_freqs.items(), key=lambda x: (x[1], x[0]))[0]
        # print("Byte pair:", byte_pair)
        merges.append(byte_pair)

        m = byte_pair[0] + byte_pair[1]
        if m not in vocab.values():
            vocab[cnt] = m
            cnt += 1

        affected = pair_index.pop(byte_pair, set())
        if not affected:
            pair_freqs[byte_pair] = 0
            continue

        new_freqs = {}
        for bchars in list(affected):
            freq = freqs.get(bchars, 0)
            if freq == 0:
                continue

            n = len(bchars)
            for i in range(n - 1):
                p = (bchars[i], bchars[i + 1])
                pair_freqs[p] = pair_freqs.get(p, 0) - freq
                s = pair_index.get(p)
                if s is not None:
                    s.discard(bchars)

            out = []
            i = 0
            while i < n:
                if i + 1 < n and bchars[i] == byte_pair[0] and bchars[i + 1] == byte_pair[1]:
                    out.append(m)
                    i += 2
                else:
                    out.append(bchars[i])
                    i += 1
            t = tuple(out)

            freqs.pop(bchars, None)
            freqs[t] = freqs.get(t, 0) + freq
            new_freqs[t] = new_freqs.get(t, 0) + freq

        for t, f in new_freqs.items():
            for i in range(len(t) - 1):
                p = (t[i], t[i + 1])
                pair_freqs[p] = pair_freqs.get(p, 0) + f
                if p not in pair_index:
                    pair_index[p] = set()
                pair_index[p].add(t)

    return vocab, merges

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    return TrainBPE(input_path, vocab_size, special_tokens)