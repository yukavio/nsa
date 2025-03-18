import torch
from torch import nn
from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_v2_func
from nsa.compression_kv import KVCompressor
from nsa.torch_attention import attention_ref as attn_func

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_v3_func
except:
    flash_attn_v3_func = None


class CompressionAttn(nn.Module):
    def __init__(self, compression_stride: int, compression_block: int, head_dim: int, flash_attn_func, device, dtype):
        super().__init__()
        self.compression_stride = compression_stride
        self.compression_block = compression_block
        self.flash_attn_func = flash_attn_func
        self.compressor = KVCompressor(compression_stride, compression_block, head_dim, device, dtype)
        

    def forward(self, **kwargs):
        k, v, cu_kv_len = self.compressor(kwargs['k'], kwargs['v'], kwargs['cu_seqlens_k'])
        q = kwargs['q']
        # kwargs['k'] = ck
        # kwargs['v'] = cv
        # max_kv_len = torch.max(cu_kv_len)
        # kwargs['cu_seqlens_k'] = cu_kv_len
        # kwargs['max_seq_len_k'] = max_kv_len
        
        bs = cu_kv_len.numel()-1
        num_q_head, head_qk_dim = q.shape[1:]
        num_token, num_kv_head, head_v_dim = v.shape
        seq_len = num_token // bs
        q = q.reshape(bs, -1, num_q_head, head_qk_dim)
        k = k.reshape(bs, seq_len, num_kv_head, head_qk_dim)
        v = v.reshape(bs, seq_len, num_kv_head, head_v_dim)
        attn_out, attn_prob = attn_func(q, k, v, self.compression_stride, self.compression_block, 
                                        causal=kwargs['causal'], scale=kwargs['softmax_scale'])
        return attn_out, attn_prob
    

class SelectionAttn(nn.Module):
    def __init__(self, selection_block: int, selected_block_count: int, flash_attn_func):
        super().__init__()
        self.selection_block = selection_block
        self.selection_block_count = selected_block_count
        self.flash_attn_func = flash_attn_func

    def forward(self, **kwargs):
        attn_score = kwargs.pop('attn')
        
        

class NSAAttention(nn.Module):
    def __init__(
        self,
        head_dim: int,
        embedding_size: int,
        causal=False,
        softmax_scale=None,
        attention_dropout=0.0,
        deterministic=False,
        compression_stride=16, # for nsa
        comression_block=32,
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
        self.gating == nn.Linear(embedding_size, 3)
        self.compression_attn = CompressionAttn(compression_stride, comression_block, head_dim, self.flash_attn_func, device, dtype)
        self.selection_attn = SelectionAttn(selection_block, selected_block_count, self.flash_attn_func)
        self.sliding_window = sliding_window
        

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        x: torch.Tensor,
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
        max_seqlen_k = max_seqlen if max_seqlen_k is None else max_seqlen_k

        # compression attn
        flash_kwargs = {
            "q": q,
            "k": k,
            "v": v,
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen,
            "max_seqlen_k": max_seqlen_k,
            "softmax_scale": self.softmax_scale,
            "causal": causal,
            "deterministic": self.deterministic,
        }
        if not self._is_v3:
            flash_kwargs["dropout_p"] = self.drop_p
        compress_attn_out, attn_score = self.compression_attn(**flash_kwargs)

        # selection attn
        flash_kwargs = {
            "q": q,
            "k": k,
            "v": v,
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen,
            "max_seqlen_k": max_seqlen_k,
            "softmax_scale": self.softmax_scale,
            "causal": causal,
            "deterministic": self.deterministic,
            'attn_score': attn_score,
        }
        if not self._is_v3:
            flash_kwargs["dropout_p"] = self.drop_p
        selection_attn_out = self.selection_attn(**flash_kwargs)

        # local attn
        flash_kwargs = {
            "q": q,
            "k": k,
            "v": v,
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen,
            "max_seqlen_k": max_seqlen_k,
            "softmax_scale": self.softmax_scale,
            "causal": causal,
            "window_size": (self.sliding_window, 0),
            "deterministic": self.deterministic,
        }
        output = self.flash_attn_func(**flash_kwargs)
        output = output[0] if self._is_v3 else output
        
        
        # v3 always returns softmax_sce
        return output

# q, k, v should before position embedding
