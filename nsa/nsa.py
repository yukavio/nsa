import torch
from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_v2_func
from torch import nn
from einops import rearrange, repeat
from nsa import selection_attention
from nsa.compression_kv import KVCompressor, KVCompressorVarlen
from nsa.triton_attention import flash_attn_func as attn_func

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_v3_func
except:
    flash_attn_v3_func = None


class NSAAttention(nn.Module):
    def __init__(
        self,
        head_dim: int,
        causal=False,
        softmax_scale=None,
        attention_dropout=0.0,
        deterministic=False,
        compression_stride=16,  # for nsa
        compression_block=32,
        selection_block=64,
        selected_block_count=16,
        sliding_window=512,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self._is_v3 = (
            head_dim % 64 == 0
            and attention_dropout == 0.0
            and flash_attn_v3_func is not None
        )
        if self._is_v3:
            print("Using FlashAttention V3")
            self.flash_attn_func = flash_attn_v3_func
        else:
            self.flash_attn_func = flash_attn_v2_func

        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop_p = attention_dropout
        self.deterministic = deterministic
        # self.compression_attn = CompressionAttn(compression_stride, compression_block, head_dim, device, dtype)
        # self.selection_attn = SelectionAttn(selection_block, selected_block_count, compression_stride, comression_block)
        self.sliding_window = sliding_window
        self.selected_block_count = selected_block_count
        self.selection_block_size = selection_block
        self.compression_stride = compression_stride
        self.compression_block = compression_block
        self.compressor = KVCompressor(
            compression_stride, compression_block, head_dim, device, dtype
        )
        kernel_size = selection_block // compression_stride + 1
        padding = compression_block // compression_stride - 2
        stride = selection_block // compression_stride
        self.pooler = torch.nn.AvgPool1d(kernel_size, stride, padding, True)
        self.gating = nn.Linear(head_dim, 3, device=device, dtype=dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        cu_seqlens_k: torch.Tensor = None,
        max_seqlen_k: torch.Tensor = None,
        causal: bool = None,
    ) -> torch.Tensor:
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value.
                If cu_seqlens is None and max_seqlen is None, then qkv has shape (B, S, 3, H, D).
                If cu_seqlens is not None and max_seqlen is not None, then qkv has shape
                (total, 3, H, D), where total is the sum of the sequence lengths in the batch.
            causal: if passed, will override self.causal
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into qkv.
            max_seqlen: int. Maximum sequence length in the batch.
        Returns:
        --------
            out: (total, H, D) if cu_seqlens is not None and max_seqlen is not None,
                else (B, S, H, D).
        """
        causal = self.causal if causal is None else causal
        cu_seqlens_k = cu_seqlens if cu_seqlens_k is None else cu_seqlens_k
        # max_seqlen_k = max_seqlen if max_seqlen_k is None else max_seqlen_k

        bs = cu_seqlens_k.numel() - 1
        num_q_head, head_qk_dim = q.shape[1:]
        num_token, num_kv_head, head_v_dim = v.shape
        q = q.reshape(bs, -1, num_q_head, head_qk_dim)  

        # compress attention
        ck, cv, compress_cu_kv_len = self.compressor(k, v, cu_seqlens_k, num_q_head//k.shape[1]) # ck/cv: B, T, H*q, D

        cmp_o, attn_score = attn_func(
            q,
            ck,
            cv,
            self.compression_stride,
            self.compression_block,
            None,
            causal,
            self.softmax_scale,
        )

        # gating
        gating_score = self.gating(q)  # b, t, hq, 3
        # selection and local attention
        score = attn_score.reshape(bs, num_kv_head, -1, *attn_score.shape[-2:]).sum(2)
        score = score.reshape(-1, *score.shape[2:])
        score = self.pooler(score)
        score = score.reshape(bs, num_kv_head, *score.shape[-2:])  # -> B, H, T1, T2
        indices = torch.topk(score, self.selected_block_count, dim=3).indices # B, H, T, S
        indices = indices.transpose(1, 2)

        k = k.reshape(bs, -1, num_kv_head, head_qk_dim)
        v = v.reshape(bs, -1, num_kv_head, head_v_dim)
        o = selection_attention(
            q,
            k,
            v,
            gating_score[..., 0],
            gating_score[..., 1],
            indices,
            None,
            self.selection_block_size,
            self.sliding_window,
            scale=self.softmax_scale,
        )

        o = torch.addcmul(o, gating_score[..., 2].unsqueeze(-1), cmp_o)

        return o.reshape(-1, *o.shape[-2:])

