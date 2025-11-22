"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""


import io
import json
import math
from dataclasses import dataclass
from typing import Literal

import blobfile as bf
import torch
import torch.nn as nn
import torch.nn.functional as F
from tiktoken.load import read_file_cached

# has to be down here to avoid loading cuda too early
from circuit_sparsity.inference.hook_utils import (
    hook_namespace,
    hook_save,
    torch_recompute_preserving_hook_context,
)
from circuit_sparsity.inference.kernels import sample_top_k


class AbsTopK(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        vals, inds = torch.topk(x.abs(), self.k, dim=-1, sorted=False)
        ret = torch.zeros_like(x)
        ret.scatter_(-1, inds, x.gather(-1, inds))
        return ret


def barrier():
    # stub
    pass


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = config.Linear(
            config.d_model, 3 * config.d_head * config.n_head, bias=config.bias
        )
        # output projection
        self.c_proj = config.Linear(config.d_head * config.n_head, config.d_model, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.dropout = config.dropout

        self.config = config
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and config.flash

        if self.flash:
            self.attn_imp = (
                SDPAWithSink(config.n_head) if config.sink else F.scaled_dot_product_attention
            )

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )
    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (d_model)

        x = self.config.maybe_activation_sparsity(x, "attn_in")
        x = hook_save("act_in", x)

        if self.config.debug_nans:
            assert x.isfinite().all(), "nan in input"

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_head * self.d_head, dim=2)

        k = self.config.maybe_activation_sparsity(k, "attn_k")
        q = self.config.maybe_activation_sparsity(q, "attn_q")
        v = self.config.maybe_activation_sparsity(v, "attn_v")

        k = hook_save("k", k)  # (B, T, n_head * d_head)
        q = hook_save("q", q)  # (B, T, n_head * d_head)
        v = hook_save("v", v)  # (B, T, n_head * d_head)

        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # (B, nh, T, hs)

        if self.config.debug_nans:
            assert q.isfinite().all(), "nan in query"
            assert k.isfinite().all(), "nan in key"
            assert v.isfinite().all(), "nan in value"

        attention_scale = 1.0 / math.sqrt(k.size(-1))

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = self.attn_imp(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
                scale=attention_scale,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * attention_scale
            att = att.masked_fill(
                self.bias[:, :, :T, :T] == 0, torch.finfo(att.dtype).min
            )  # float("-inf"))

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        if self.config.debug_nans:
            assert y.isfinite().all(), "nan in attention output"

        y = (
            y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.d_head)
        )  # re-assemble all head outputs side by side

        # y = self.config.maybe_activation_sparsity(y)
        y = hook_save("y", y)  # (B, T, n_head * d_head)

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        if self.config.debug_nans:
            assert y.isfinite().all(), "nan in attention output 2"

        y = self.config.maybe_activation_sparsity(y, "attn_out")
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_fc = config.Linear(config.d_model, config.d_mlp, bias=config.bias)
        self.act_fn = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
        }[config.activation_type]
        self.c_proj = config.Linear(config.d_mlp, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.config.maybe_activation_sparsity(x, "mlp_in")
        x = hook_save("act_in", x)

        if self.config.debug_nans:
            assert x.isfinite().all(), "nan in mlp input"

        x = self.c_fc(x)

        if self.config.debug_nans:
            assert x.isfinite().all(), "nan in mlp after c_fc"

        x = self.act_fn(x)
        x = self.config.maybe_activation_sparsity(x, "mlp_neuron")
        x = hook_save("post_act", x)

        if self.config.debug_nans:
            assert x.isfinite().all(), "nan in mlp after act"

        x = self.c_proj(x)

        if self.config.debug_nans:
            assert x.isfinite().all(), "nan in mlp after c_proj"
        x = self.dropout(x)

        x = self.config.maybe_activation_sparsity(x, "mlp_out")
        return x


class SDPAWithSink(nn.Module):
    """
    Adds a learnable denominator-only term ("attention sink") to SDPA by
    concatenating a dummy KV slot whose logit is b and whose V is zero.
    """

    def __init__(self, num_heads: int, init_logit: float = 0.0):
        super().__init__()
        shape = (num_heads,)
        self.sink_logit = nn.Parameter(torch.full(shape, init_logit))

    def forward(
        self,
        q: torch.Tensor,  # (B, H, Lq, D)
        k: torch.Tensor,  # (B, H, Lk, D)
        v: torch.Tensor,  # (B, H, Lk, Dv)
        *,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
    ) -> torch.Tensor:
        B, H, Lq, D = q.shape
        _, _, Lk, _ = k.shape
        Dv = v.size(-1)

        # 1) Prepend a dummy KV slot (always visible)
        k_sink = torch.zeros((B, H, 1, D), dtype=q.dtype, device=q.device)
        v_sink = torch.zeros((B, H, 1, Dv), dtype=v.dtype, device=v.device)
        k_aug = torch.cat([k_sink, k], dim=2)  # (B,H,Lk+1,D)
        v_aug = torch.cat([v_sink, v], dim=2)  # (B,H,Lk+1,Dv)

        # 2) Build shifted causal allow-mask over keys (columns 1..), always allow col 0 (sink)
        # allow: 1 where attending is allowed, 0 where disallowed
        # For real keys: allow[i, j+1] = 1 if j <= i else 0  (lower-triangular)
        allow = torch.zeros((Lq, Lk + 1), dtype=torch.bool, device=q.device)
        allow[:, 0] = True  # sink column always on
        # lower-triangular for real keys shifted by +1
        real = torch.ones((Lq, Lk), dtype=torch.bool, device=q.device).tril()
        allow[:, 1:] = real

        # Broadcast to (B,H,Lq,Lk+1)
        allow = allow.view(1, 1, Lq, Lk + 1).expand(B, H, Lq, Lk + 1)

        # 3) Turn it into an additive mask. 0 for allowed, -inf for disallowed
        neg_inf = torch.finfo(q.dtype).min
        base_mask = torch.where(
            allow,
            torch.zeros((), dtype=q.dtype, device=q.device),
            torch.full((), neg_inf, dtype=q.dtype, device=q.device),
        )  # (B,H,Lq,Lk+1)

        # 4) Add learnable sink bias b to column 0 (per head or shared)
        if self.sink_logit.numel() == H:
            b = self.sink_logit.to(dtype=q.dtype, device=q.device).view(1, H, 1, 1)  # (1,H,1,1)
        else:
            b = self.sink_logit.to(dtype=q.dtype, device=q.device).view(1, 1, 1, 1)  # (1,1,1,1)

        sink_bias_mask = torch.zeros((1, 1, 1, Lk + 1), dtype=q.dtype, device=q.device)
        sink_bias_mask[..., 0] = 1.0
        attn_mask = base_mask + sink_bias_mask * b  # (B,H,Lq,Lk+1)

        # 5) SDPA with our custom mask; keep is_causal=False to avoid double-masking
        out = F.scaled_dot_product_attention(
            q,
            k_aug,
            v_aug,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,  # important
            scale=scale,
        )
        return out


class Block(nn.Module):
    # block exactly satisfies the invariant that forward = forward_mlp_block . forward_attn_block
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.ln_1 = (
            nn.RMSNorm(config.d_model)
            if config.rms_norm
            else LayerNorm(config.d_model, bias=config.ln_bias)
        )
        self.attn = CausalSelfAttention(config)
        self.ln_2 = (
            nn.RMSNorm(config.d_model)
            if config.rms_norm
            else LayerNorm(config.d_model, bias=config.ln_bias)
        )
        self.mlp = MLP(config)

    def forward_attn_block(self, x):
        x = hook_save("resid_in", x)

        if self.config.debug_nans:
            assert x.isfinite().all(), "nan in blk input"

        with hook_namespace("attn"):
            if self.config.grad_checkpointing:
                x = x + hook_save(
                    "resid_delta",
                    torch_recompute_preserving_hook_context(
                        lambda x: self.attn(self.ln_1(x)), x, use_reentrant=False
                    ),
                )
            else:
                x = x + hook_save("resid_delta", self.attn(self.ln_1(x)))

        if self.config.residual_activation_type == "relu":
            x = torch.relu(x)
        x = self.config.maybe_activation_sparsity(x, "resid_post_attn")

        return x

    def forward_mlp_block(self, x):
        x = hook_save("resid_mid", x)
        with hook_namespace("mlp"):
            if self.config.grad_checkpointing:
                x = x + hook_save(
                    "resid_delta",
                    torch_recompute_preserving_hook_context(
                        lambda x: self.mlp(self.ln_2(x)), x, use_reentrant=False
                    ),
                )
            else:
                x = x + hook_save("resid_delta", self.mlp(self.ln_2(x)))

        if self.config.residual_activation_type == "relu":
            x = torch.relu(x)
        x = self.config.maybe_activation_sparsity(x, "resid_post_mlp")
        return x

    def forward(self, x):
        x = self.forward_attn_block(x)
        x = self.forward_mlp_block(x)
        return x


class CausalSelfAttentionCatPosEmb(CausalSelfAttention):
    def __init__(self, config):
        # initialize base attention with standard shapes, we'll override projections
        super().__init__(config)
        assert config.d_model % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = config.Linear(
            config.d_model_in, 3 * config.d_head * config.n_head, bias=config.bias
        )
        # output projection
        self.c_proj = config.Linear(config.d_head * config.n_head, config.d_model, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model_in = config.d_model_in
        self.d_model = config.d_model
        self.dropout = config.dropout
        self.config = config
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and config.flash

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )
    def forward(self, x, pos_emb_to_cat):
        # Broadcast pos emb over batch if provided as shape [1, T, C]
        if pos_emb_to_cat is not None and pos_emb_to_cat.size(0) == 1 and x.size(0) != 1:
            pos_emb_to_cat = pos_emb_to_cat.expand(x.size(0), -1, -1)
        x = torch.cat([x, pos_emb_to_cat], dim=-1)
        return super().forward(x)


class MLPCatPosEmb(MLP):
    def __init__(self, config):
        # initialize base MLP, we'll override the projections to match cat shapes
        super().__init__(config)
        self.config = config
        self.c_fc = config.Linear(config.d_model_in, config.d_mlp, bias=config.bias)
        self.act_fn = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
        }[config.activation_type]
        self.c_proj = config.Linear(config.d_mlp, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, pos_emb_to_cat):
        # Broadcast pos emb over batch if provided as shape [1, T, C]
        if pos_emb_to_cat is not None and pos_emb_to_cat.size(0) == 1 and x.size(0) != 1:
            pos_emb_to_cat = pos_emb_to_cat.expand(x.size(0), -1, -1)
        x = torch.cat([x, pos_emb_to_cat], dim=-1)
        x = super().forward(x)
        return x


class BlockCatPosEmb(Block):
    # block exactly satisfies the invariant that forward = forward_mlp_block . forward_attn_block
    def __init__(self, config):
        # initialize base Block to get ln_1/ln_2 and other invariants
        super().__init__(config)
        self.ln_p1 = (
            nn.RMSNorm(config.d_pos_emb)
            if config.rms_norm
            else LayerNorm(config.d_pos_emb, bias=config.ln_bias)
        )
        self.ln_p2 = (
            nn.RMSNorm(config.d_pos_emb)
            if config.rms_norm
            else LayerNorm(config.d_pos_emb, bias=config.ln_bias)
        )
        self.attn = CausalSelfAttentionCatPosEmb(config)
        self.mlp = MLPCatPosEmb(config)

    def forward_attn_block(self, x, p):
        x = hook_save("resid_in", x)

        if self.config.debug_nans:
            assert x.isfinite().all(), "nan in blk input"

        with hook_namespace("attn"):
            if self.config.grad_checkpointing:
                x = x + hook_save(
                    "resid_delta",
                    torch_recompute_preserving_hook_context(
                        lambda x, p: self.attn(self.ln_1(x), self.ln_p1(p)),
                        x,
                        p,
                        use_reentrant=False,
                    ),
                )
            else:
                x = x + hook_save("resid_delta", self.attn(self.ln_1(x), self.ln_p1(p)))

        if self.config.residual_activation_type == "relu":
            x = torch.relu(x)
        x = self.config.maybe_activation_sparsity(x, "resid_post_attn")

        return x

    def forward_mlp_block(self, x, p):
        x = hook_save("resid_mid", x)
        with hook_namespace("mlp"):
            if self.config.grad_checkpointing:
                x = x + hook_save(
                    "resid_delta",
                    torch_recompute_preserving_hook_context(
                        lambda x, p: self.mlp(self.ln_2(x), self.ln_p2(p)),
                        x,
                        p,
                        use_reentrant=False,
                    ),
                )
            else:
                x = x + hook_save("resid_delta", self.mlp(self.ln_2(x), self.ln_p2(p)))

        if self.config.residual_activation_type == "relu":
            x = torch.relu(x)
        x = self.config.maybe_activation_sparsity(x, "resid_post_mlp")
        return x

    def forward(self, x, pos_emb_to_cat):
        x = self.forward_attn_block(x, pos_emb_to_cat)
        x = self.forward_mlp_block(x, pos_emb_to_cat)
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency  # TODO: FLAG FOR ACHY
    n_layer: int = 12
    n_head: int = 12
    d_head: int | None = None  # defaults to d_model // n_head
    d_model: int = 768
    dropout: float = 0.0
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )
    ln_bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )
    rms_norm: bool = False  # use RMSNorm instead of LayerNorm
    residual_activation_type: Literal["identity", "relu"] = "identity"
    activation_type: Literal["gelu", "relu"] = "gelu"
    afrac: float | None = None  # fraction of activations to keep
    afrac_loctypes: str = "attn_in,attn_out,mlp_in,mlp_out"
    debug_nans: bool = False
    tied_unembed: bool = True

    tokenizer_name: str = "tinypython_2k"

    grad_checkpointing: bool = True
    d_mlp: int | None = None  # multiplier for MLP hidden layer size

    enable_bigram_table: bool = False
    learnable_bigram_table: bool = False
    d_pos_emb: int | None = None
    dropout_cat_pos_emb: bool = False
    sinusoidal_cat_pos_emb: bool = False
    enable_sparse_kernels: bool = False

    flash: bool = True
    sink: bool = False

    @property
    def cat_pos_emb(self):
        return self.d_pos_emb is not None

    @property
    def d_model_in(self):
        return self.d_model + self.d_pos_emb if self.cat_pos_emb else self.d_model

    def __post_init__(self):
        assert self.d_model % self.n_head == 0
        assert self.residual_activation_type in ["identity", "relu"]
        assert self.activation_type in ["gelu", "relu"]

        if self.d_mlp is None:
            self.d_mlp = 4 * self.d_model
        if self.d_head is None:
            self.d_head = self.d_model // self.n_head

    @property
    def Linear(self):
        return nn.Linear

    def maybe_activation_sparsity(self, x, loctype):
        if self.afrac is not None and loctype in self.afrac_loctypes.split(","):
            def keep_abstopk(x, k):
                ret = torch.zeros_like(x)
                _, topk_inds = torch.topk(x.abs(), k, dim=-1, sorted=False)
                ret.scatter_(-1, topk_inds, x.gather(-1, topk_inds))
                return ret

            x = keep_abstopk(
                x,
                k=int(self.afrac * x.shape[-1]),
            )

        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        if config.cat_pos_emb:
            block_cls = BlockCatPosEmb
        else:
            block_cls = Block

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.d_model),
                wpe=nn.Embedding(config.block_size, config.d_pos_emb or config.d_model),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([(block_cls(config)) for _ in range(config.n_layer)]),
                ln_f=nn.RMSNorm(config.d_model)
                if config.rms_norm
                else LayerNorm(config.d_model, bias=config.ln_bias),
            )
        )

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.register_buffer(
            "final_logits_bias", torch.zeros(config.vocab_size, dtype=torch.float32)
        )

        if self.config.enable_bigram_table:
            if self.config.learnable_bigram_table:
                # HACK: low rank to fit in mem
                self.bigram_table = nn.Parameter(
                    torch.zeros(config.vocab_size, config.vocab_size, dtype=torch.float32)
                )
            else:
                self.register_buffer(
                    "bigram_table",
                    torch.zeros(config.vocab_size, config.vocab_size, dtype=torch.float32),
                )
        else:
            self.bigram_table = None

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        if config.tied_unembed:
            self.transformer.wte.weight = (
                self.lm_head.weight
            )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                if p.is_sparse:
                    num_nonzero = p._nnz()
                    p._values().data = (
                        sample_top_k(n=p.numel(), k=num_nonzero, shape=(num_nonzero,))
                        * 0.02
                        / math.sqrt(2 * config.n_layer)
                    )
                else:
                    torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # If requested, initialize positional embeddings with fixed sinusoids and freeze
        if config.cat_pos_emb and config.sinusoidal_cat_pos_emb:
            assert config.d_pos_emb is not None, (
                "sinusoidal_cat_pos_emb requires cat_pos_emb (d_pos_emb must be set)"
            )
            with torch.no_grad():
                T = config.block_size
                D = config.d_pos_emb
                device = self.transformer.wpe.weight.device
                dtype = self.transformer.wpe.weight.dtype
                positions = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)  # [T,1]
                d_half = max(1, D // 2)
                # periods from 4 tokens up to block_size tokens (log-spaced)
                T_float = float(T)
                p_min = 4.0
                p_max = max(p_min, T_float)
                periods = torch.logspace(
                    math.log10(p_min), math.log10(p_max), steps=d_half, device=device, dtype=dtype
                )
                freqs = 2 * math.pi / periods  # [d_half]
                angles = positions * freqs  # [T, d_half]
                sinv = torch.sin(angles)
                cosv = torch.cos(angles)
                enc = torch.cat([sinv, cosv], dim=1)  # [T, 2*d_half]
                if enc.shape[1] < D:
                    pad = torch.zeros(T, D - enc.shape[1], device=device, dtype=dtype)
                    enc = torch.cat([enc, pad], dim=1)
                elif enc.shape[1] > D:
                    enc = enc[:, :D]
                self.transformer.wpe.weight.copy_(enc)
                self.transformer.wpe.weight.requires_grad_(False)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, include_resid_mid=False):
        device = idx.device
        b, t = idx.size()

        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )
        # pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, d_model)
        # pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, d_model)
        pos_emb = self.transformer.wpe.weight[:t].unsqueeze(0)
        if self.config.cat_pos_emb:
            x = self.transformer.drop(tok_emb)
        else:
            x = self.transformer.drop(tok_emb + pos_emb)

        if self.config.debug_nans:
            assert x.isfinite().all(), "nan in initial post-embedding"

        if self.config.enable_bigram_table:
            # add bigram table to the logits bias
            additional_logits_bias = F.embedding(idx, self.bigram_table, padding_idx=-1)
            additional_logits_bias = additional_logits_bias.to(x.dtype)
        else:
            additional_logits_bias = None

        if self.config.cat_pos_emb:
            pos_emb_to_cat = pos_emb
            if self.config.dropout_cat_pos_emb:
                pos_emb_to_cat = self.transformer.drop(pos_emb)
        else:
            pos_emb_to_cat = None

        return self.forward_tail(
            x,
            n=0,
            targets=targets,
            additional_logits_bias=additional_logits_bias,
            include_resid_mid=include_resid_mid,  # this is hacky we should just switch to using hooks
            pos_emb_to_cat=pos_emb_to_cat,
        )

    def forward_tail(
        self,
        x,
        n,
        targets=None,
        additional_logits_bias=None,
        include_resid_mid=False,
        pos_emb_to_cat=None,
    ):
        # print(x.shape)
        hs = []
        blks = list(self.transformer.h)

        if include_resid_mid:
            blks = list_join(
                [
                    [
                        blk.forward_attn_block,
                        blk.forward_mlp_block,
                    ]
                    for blk in blks
                ]
            )

        assert n <= len(blks)

        for i, block_fn in enumerate(blks[n:]):
            global curlayer
            curlayer = i
            with hook_namespace(f"{i // 2}") if include_resid_mid else hook_namespace(f"{i}"):
                hs.append(x)
                if self.config.cat_pos_emb:
                    x = block_fn(x, pos_emb_to_cat)
                else:
                    x = block_fn(x)

        x = hook_save("final_resid", x)
        x = self.transformer.ln_f(x)

        logits = (
            self.lm_head(x)
            + self.final_logits_bias
            + (additional_logits_bias if additional_logits_bias is not None else 0)
        )
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            loss = torch.zeros(1, device=x.device)

        return logits, loss, hs  # hs is deprecated in favor of hook stuff

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def is_mlp_param(self, p):
        return id(p) in list_join(
            [
                [
                    id(self.transformer.h[i].mlp.c_fc.weight),
                    id(self.transformer.h[i].mlp.c_proj.weight),
                ]
                for i in range(self.config.n_layer)
            ]
        )

    def is_param_embed(self, p):
        return p is self.transformer.wte.weight or p is self.transformer.wpe.weight

    def is_attn_param(self, p):
        return id(p) in list_join(
            [
                [
                    id(self.transformer.h[i].attn.c_attn.weight),
                    id(self.transformer.h[i].attn.c_proj.weight),
                ]
                for i in range(self.config.n_layer)
            ]
        )

    def is_bias(self, p):
        return id(p) in list_join(
            [
                [
                    id(self.transformer.h[i].attn.c_attn.bias),
                    id(self.transformer.h[i].attn.c_proj.bias),
                    id(self.transformer.h[i].mlp.c_fc.bias),
                    id(self.transformer.h[i].mlp.c_proj.bias),
                ]
                for i in range(self.config.n_layer)
            ]
        )

    def is_ln_param(self, p):
        return id(p) in list_join(
            [
                [
                    id(self.transformer.h[i].ln_1.weight),
                    id(self.transformer.h[i].ln_2.weight),
                ]
                for i in range(self.config.n_layer)
            ]
        ) + [
            id(self.transformer.ln_f.weight),
        ]

    def is_sparse_param(self, p, dense_embeddings=None, dense_unembed=None, dense_biases=None):
        # if these params aren't specified, then still give answers, but only for uncontroversial params

        if dense_embeddings is None:
            assert p is not self.transformer.wte.weight and p is not self.transformer.wpe.weight
        if dense_unembed is None:
            assert p is not self.lm_head.weight
        if dense_biases is None:
            assert not self.is_bias(p)

        if p is self.transformer.wte.weight or p is self.transformer.wpe.weight:
            return not dense_embeddings
        if p is self.lm_head.weight:
            return not dense_unembed
        if self.is_bias(p):
            return not dense_biases

        return id(p) in list_join(
            [
                [
                    id(self.transformer.h[i].attn.c_attn.weight),
                    id(self.transformer.h[i].attn.c_proj.weight),
                    id(self.transformer.h[i].mlp.c_fc.weight),
                    id(self.transformer.h[i].mlp.c_proj.weight),
                ]
                for i in range(self.config.n_layer)
            ]
        )


def list_join(xss: list[list]) -> list:
    """monadic join for lists"""
    return [x for xs in xss for x in xs]


def load_model(model_path, flash=False, grad_checkpointing=False, cuda=True):
    beeg_config_json = json.loads(
        read_file_cached(f"{model_path}/beeg_config.json").decode()
    )
    if "n_mlp" in beeg_config_json:
        beeg_config_json["d_mlp"] = beeg_config_json.pop("n_mlp")
    beeg_config_json["flash"] = flash
    if "sink" not in beeg_config_json:
        beeg_config_json["sink"] = False
    assert not (beeg_config_json["sink"] and not flash), "WARNING: attention sink is only supported with flash=True, this will silently break the model"
    beeg_config_json["grad_checkpointing"] = grad_checkpointing

    if "use_tied_aux_matrix" in beeg_config_json:
        assert not beeg_config_json.pop("use_tied_aux_matrix")

    config = GPTConfig(**beeg_config_json)
    ckpt_path = bf.join(f"{model_path}", "final_model.pt")

    model = GPT(config)
    # model.bake_tied_aux_matrix_()
    map_location = "cuda" if cuda else "cpu"
    sd = torch.load(
        io.BytesIO(read_file_cached(ckpt_path)),
        weights_only=True,
        map_location=map_location,
    )
    if "final_logits_bias" not in sd:
        sd["final_logits_bias"] = torch.zeros(config.vocab_size)
    model.load_state_dict(sd, strict=False)

    if cuda:
        model.cuda()
    model.eval()
    return model
