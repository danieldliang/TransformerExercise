from __future__ import annotations
import typing
import torch
from torch import Tensor
import math
from jaxtyping import Bool, Float, Int
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
    
class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        angles = torch.empty(max_seq_len, d_k // 2, dtype=torch.float32)
        for i in range(max_seq_len):
            for k in range(d_k // 2):
                angles[i][k] = i / (theta ** ((2 * k - 2) / d_k))
        
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        original = x.dtype
        x = x.to(torch.float32)
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

def SoftMax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    # Edge case: negative infinity mask is applied to the entire row
    x = torch.where(in_features != -torch.inf, in_features - torch.amax(in_features, dim=dim, keepdim=True), -torch.inf)
    # In case negative infinity mask is applied to the entire row, we need to check that the sum is strictly greater than 0 
    return torch.where(torch.sum(torch.exp(x), dim=dim, keepdim=True) > 0, torch.exp(x) / torch.sum(torch.exp(x), dim=dim, keepdim=True), 0)

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

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float = 1009, max_seq_len: int=2048):
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
        Q = rearrange(QKV[..., 0,:,:,:], " ... seq_len num_heads d_k -> ... num_heads seq_len d_k")
        K = rearrange(QKV[..., 1,:,:,:], " ... seq_len num_heads d_k -> ... num_heads seq_len d_k")
        V = rearrange(QKV[..., 3,:,:,:], " ... seq_len num_heads d_k -> ... num_heads seq_len d_k")

        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)
        mask = torch.empty((seq_len, seq_len), dtype=bool)
        for i in range(seq_len):
            for j in range(0, i):
                mask[i][j] = False
            for j in range(i, seq_len):
                mask[i][j] = True
        
        multihead = scaled_dot_product_attention(Q, K, V, mask)
        return self.W_O(multihead)

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