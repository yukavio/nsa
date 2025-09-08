import torch
from torch import nn
import torch.nn.functional as f
from einops import rearrange, repeat
from nsa import selection_attention
from nsa.compression_kv import KVCompressor, KVCompressorVarlen
from nsa.triton_attention import flash_attn_func as attn_func
from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_v2_func

try:
    from flash_attn_interface import flash_attn_func as flash_attn_v3_func
except:
    flash_attn_v3_func = None


class NSAFunctor:
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
        seperated_kv=False,
    ):
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
        self.sliding_window = sliding_window
        self.selected_block_count = selected_block_count
        self.selection_block_size = selection_block
        self.compression_stride = compression_stride
        self.compression_block = compression_block
        self.pool_kernel_size = selection_block // compression_stride + 1
        self.pool_padding = compression_block // compression_stride - 2
        self.pool_stride = selection_block // compression_stride
        self.seperated_kv = seperated_kv

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        compressor_weight_k: torch.Tensor,
        compressor_weight_v: torch.Tensor,
        gating_weight: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: torch.Tensor = None,
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
        # compress attention
        bs = q.shape[0] if cu_seqlens_k is None else cu_seqlens_k.numel() - 1
        num_q_head, head_qk_dim = q.shape[-2:]
        num_kv_head, head_v_dim = v.shape[-2:]
        q = q.reshape(bs, -1, num_q_head, head_qk_dim)  
        gating_score = f.linear(q, gating_weight)  # b, t, hq, 3
        k = k.reshape(bs, -1, num_kv_head, head_qk_dim)
        v = v.reshape(bs, -1, num_kv_head, head_v_dim)

        if self.seperated_kv:
            assert num_kv_head % 3 == 0
            num_kv_head = num_kv_head // 3
            
            # for compression attn
            ck = k[:, :, :num_kv_head, :].permute(0, 2, 3, 1).reshape(bs*num_kv_head, head_qk_dim, -1)
            cv = v[:, :, :num_kv_head, :].permute(0, 2, 3, 1).reshape(bs*num_kv_head, head_v_dim, -1)
            # for selection attn
            select_k = k[:, :, num_kv_head:num_kv_head*2, :].contiguous()
            select_v = v[:, :, num_kv_head:num_kv_head*2, :].contiguous()
            # sliding window
            sliding_k = k[:, :, num_kv_head*2:, :].contiguous()
            sliding_v = v[:, :, num_kv_head*2:, :].contiguous()
        else:
            ck = k.permute(0, 2, 3, 1).reshape(bs*num_kv_head, head_qk_dim, -1)
            cv = v.permute(0, 2, 3, 1).reshape(bs*num_kv_head, head_v_dim, -1)
            # for selection attn
            select_k = k
            select_v = v
            # sliding window
            sliding_k = k
            sliding_v = v
        ck = f.conv1d(ck, compressor_weight_k, stride=self.compression_stride, padding=0)
        cv = f.conv1d(cv, compressor_weight_v, stride=self.compression_stride, padding=0)
        ck = ck.reshape(bs, num_kv_head, head_qk_dim, -1).permute(0, 3, 1, 2)
        cv = cv.reshape(bs, num_kv_head, head_v_dim, -1).permute(0, 3, 1, 2)
        ck = repeat(ck, "b s h d -> b s (h g) d", g=num_q_head//num_kv_head).contiguous()
        cv = repeat(cv, "b s h d -> b s (h g) d", g=num_q_head//num_kv_head).contiguous()
        
        cmp_o, indices = attn_func(
            q,
            ck,
            cv,
            self.compression_stride,
            self.compression_block,
            causal,
            self.softmax_scale,
            num_kv_head,
            self.pool_kernel_size,
            self.pool_stride,
            self.pool_padding,
            self.selected_block_count,
        )
        o = selection_attention(
                q,
                select_k,
                select_v,
                gating_score[..., 0],
                gating_score[..., 1],
                indices,
                None,
                self.selection_block_size,
                0 if self.seperated_kv else self.sliding_window,
                scale=self.softmax_scale,
            )
        
        if self.seperated_kv:
            sliding_o = flash_attn_v3_func(q, sliding_k, sliding_v, softmax_scale=self.softmax_scale, causal=self.causal, 
                                           window_size=(self.sliding_window-1, 0))[0]
            o = torch.addcmul(o, gating_score[..., 1].unsqueeze(-1), sliding_o)
        o = torch.addcmul(o, gating_score[..., 2].unsqueeze(-1), cmp_o)
        return o.reshape(-1, *o.shape[-2:])


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
        self.nsa_functor = NSAFunctor(head_dim, causal, softmax_scale, attention_dropout, deterministic,
                                 compression_stride, compression_block, selection_block, selected_block_count,
                                 sliding_window, device, dtype)
        self.compressor = KVCompressor(
            compression_stride, compression_block, head_dim, device, dtype
        )
        self.gating = nn.Linear(head_dim, 3, device=device, dtype=dtype)
        
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor = None,
        cu_seqlens_k: torch.Tensor = None,
        max_seqlen_k: torch.Tensor = None,
        causal: bool = None,
    ) -> torch.Tensor:
        cu_seqlens_k = cu_seqlens if cu_seqlens_k is None else cu_seqlens_k
        # compress attention
        num_q_head, head_qk_dim = q.shape[1:]
        ck, cv, compress_cu_kv_len = self.compressor(k, v, cu_seqlens_k, num_q_head//k.shape[1]) # ck/cv: B, T, H*q, D
        return self.nsa_functor.forward(q, k, v, ck, cv, self.gating.weight, cu_seqlens, max_seqlen, causal=causal)
        
        
# Demo, For Intergration test
class MergedNSAAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        nsa_functor: NSAFunctor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gating_weight: torch.Tensor,
        compressor_weight_k: torch.Tensor,
        compressor_weight_v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor = None,
        cu_seqlens_k: torch.Tensor = None,
        max_seqlen_k: torch.Tensor = None,
        causal: bool = None,
    ) -> torch.Tensor:
        with torch.enable_grad():
            o = nsa_functor.forward(q, k, v, compressor_weight_k, compressor_weight_v, 
                                    gating_weight, cu_seqlens, max_seqlen,cu_seqlens_k, 
                                    max_seqlen_k, causal)
        ctx.save_for_backward(q, k, v, gating_weight, compressor_weight_k, 
                                compressor_weight_v, o)
        return o.clone()
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, gating_weight, cmp_w_k, cmp_w_v, o = ctx.saved_tensors
        
        q_grad, k_grad, v_grad, gating_grad, cmp_k_grad, cmp_v_grad = torch.autograd.grad(
                outputs=(o),
                inputs=(q, k, v, gating_weight, cmp_w_k, cmp_w_v),
                grad_outputs=do,
            )
        return None, q_grad, k_grad, v_grad, gating_grad, cmp_k_grad, cmp_v_grad, None, None, None, None, None,
