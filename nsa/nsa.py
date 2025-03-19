import torch
from torch import nn
from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_v2_func
from nsa.compression_kv import KVCompressor
from nsa.torch_attention import attention_ref as attn_func
from nsa import selection_attention

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_v3_func
except:
    flash_attn_v3_func = None


class CompressionAttn(nn.Module):
    def __init__(self, compression_stride: int, compression_block: int, head_dim: int, device, dtype):
        super().__init__()
        self.compression_stride = compression_stride
        self.compression_block = compression_block
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
    def __init__(self, selection_block: int, selected_block_count: int, 
                 compression_stride: int, compression_block: int):
        super().__init__()
        self.selection_block = selection_block
        self.selection_block_count = selected_block_count
        
        kernel_size = selection_block // compression_stride + 1
        padding = compression_block // compression_stride - 2
        stride = selection_block // compression_stride
        self.pooler = torch.nn.AvgPool1d(kernel_size, stride, padding, True)

    def forward(self, **kwargs):
        attn_score = kwargs.pop('attn')
        q = kwargs['q']
        k = kwargs['k']
        v = kwargs['v']
        
        bs, seq_len = attn_score.shape[:2]
        num_token, num_head, head_dim = q.shape
        num_kv_head = k.shape[1]
        
        score = attn_score.reshape(bs, num_kv_head, -1, *attn_score.shape[-2:]).sum(2)
        score = score.reshape(-1, *score.shape[2:])
        score = self.pooler(score).reshape(bs, seq_len, score.shape[-2:]) # -> B, H, T1, T2
        
        indices = torch.topk(score, self.selection_block_count, dim=3).indices
        return bs
        

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
        #self.compression_attn = CompressionAttn(compression_stride, compression_block, head_dim, device, dtype)
        #self.selection_attn = SelectionAttn(selection_block, selected_block_count, compression_stride, comression_block)
        self.sliding_window = sliding_window
        self.selected_block_count = selected_block_count
        self.selection_block_size = selection_block
        self.compressor = KVCompressor(compression_stride, compression_block, head_dim, device, dtype)
        kernel_size = selection_block // compression_stride + 1
        padding = compression_block // compression_stride - 2
        stride = selection_block // compression_stride
        self.pooler = torch.nn.AvgPool1d(kernel_size, stride, padding, True)
        self.gating = nn.Linear(head_dim, 3)

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

        # compress attention
        ck, cv, compress_cu_kv_len = self.compressor(k, v, cu_seqlens_k)
        bs = cu_seqlens_k.numel()-1
        num_q_head, head_qk_dim = q.shape[1:]
        num_token, num_kv_head, head_v_dim = v.shape
        seq_len = num_token // bs
        q = q.reshape(bs, -1, num_q_head, head_qk_dim)
        k = k.reshape(bs, seq_len, num_kv_head, head_qk_dim)
        v = v.reshape(bs, seq_len, num_kv_head, head_v_dim)
        cmp_o, attn_score = attn_func(q, k, v, self.compression_stride, self.compression_block, 
                                            causal=causal, scale=self.softmax_scale)
        
        # gating
        gating_score = self.gating(q) # b, hq, t, 3
        
        # selection and local attention
        compress_seq_len = attn_score.shape[1]
        score = attn_score.reshape(bs, num_kv_head, -1, *attn_score.shape[-2:]).sum(2)
        score = score.reshape(-1, *score.shape[2:])
        score = self.pooler(score).reshape(bs, num_kv_head, score.shape[-2:]) # -> B, H, T1, T2
        indices = torch.topk(score, self.selection_block_count, dim=3).indices

        o = selection_attention(q, k, v, self.gating[...,0], self.gating[...,1], indices, self.selected_block_count, 
                            self.selection_block_size, self.sliding_window, scale=self.softmax_scale)
        
        o = torch.addcmul(o, self.gating[..., 2], cmp_o)

        return o
