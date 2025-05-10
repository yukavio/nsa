import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange, repeat
from torch.nn.attention import SDPBackend, sdpa_kernel


# Copy from https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py
def construct_local_mask(
    seqlen_q,
    seqlen_k,
    block_stride,
    block_size,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )

    if window_size[0] < 0:
        return col_idx * block_stride + block_size > row_idx
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        mask = torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )
        return mask


def get_indices(s, pool_num_kv_head=0, pool_kernel_size=0, pool_stride=0, 
                pool_padding=0, select_block_count=0):
    bs = s.shape[0]
    s = s.reshape(bs, pool_num_kv_head, -1, *s.shape[-2:]).sum(2)
    s = s.reshape(-1, *s.shape[2:])
    s = torch.nn.functional.avg_pool1d(s, pool_kernel_size, pool_stride, 
                                    pool_padding, True)
    s = s.reshape(bs, pool_num_kv_head, *s.shape[-2:])  # -> B, H, T1, T2
    indices = torch.topk(s, select_block_count, dim=3).indices # B, H, T1, S
    indices = indices.transpose(1, 2).contiguous()
    return indices



# Copy from https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py
def attention_ref(
    q,
    k,
    v,
    block_stride, 
    block_size,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=False,
    reorder_ops=False,
    key_leftpad=None,
    scale=None,
    pool_num_kv_head=0,
    pool_kernel_size=0,
    pool_stride=0,
    pool_padding=0,
    select_block_count=0
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    d = q.shape[-1]
    if scale is None:
        scale = 1 / math.sqrt(d)
    if not reorder_ops:
        qk = torch.einsum("bthd,bshd->bhts", q, k)
    else:
        qk = torch.einsum("bthd,bshd->bhts", q, k)
    compress_score = torch.softmax(qk, dim=-1)
    indicis = get_indices(compress_score, pool_num_kv_head, pool_kernel_size, pool_stride, pool_padding, select_block_count)
    scores = qk * scale

    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            block_stride,
            block_size,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=key_leftpad,
        )
        if causal:
            scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    # scores.retain_grad()
    attention_without_mask = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if (window_size[0] >= 0 or window_size[1] >= 0) and causal:
        attention = attention_without_mask.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    else:
        attention = attention_without_mask

    # attention_without_mask.retain_grad()
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), indicis

