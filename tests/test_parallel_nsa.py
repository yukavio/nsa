# This script is used to test the parallel_nsa function.

# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
# ruff: noqa
from typing import Optional

import torch
from typing import Union
from einops import rearrange, repeat
from nsa import selection_attention

def naive_nsa(q: torch.Tensor,
              k: torch.Tensor,
              v: torch.Tensor,
              g_slc: torch.Tensor,
              g_swa: torch.Tensor,
              block_indices: torch.LongTensor,
              block_counts: Optional[Union[torch.LongTensor, int]] = None,
              block_size: int = 64,
              window_size: int = 0,
              scale: Optional[float] = None,
              cu_seqlens: Optional[torch.LongTensor] = None,
              head_first: bool = False) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, HQ, K]` if `head_first=False` else `[B, HQ, T, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
            GQA is enforced here. The ratio of query heads (HQ) to key/value heads (H) must be a power of 2 and >=16.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        g_slc (torch.Tensor):
            Gate score for selected attention of shape `[B, T, HQ]` if  `head_first=False` else `[B, HQ, T]`.
        g_swa (torch.Tensor):
            Gate score for sliding attentionof shape `[B, T, HQ]` if  `head_first=False` else `[B, HQ, T]`.
        block_indices (torch.LongTensor):
            Block indices of shape `[B, T, H, S]` if `head_first=False` else `[B, H, T, S]`.
            `S` is the maximum number of selected blocks for each query token, which is set to 16 in the paper.
        block_counts (Union[torch.LongTensor, int]):
            Number of selected blocks for each token.
            If a tensor is provided, with shape `[B, T, H]` if `head_first=True` else `[B, T, H]`,
            each token can select the same number of blocks.
            If not provided, it will default to `S`, Default: `None`.
        block_size (int):
            Selected block size. Default: 64.
        window_size (int):
            Sliding window size. Default: 0.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]` if `head_first=False` else `[B, HQ, T, V]`.
    """
    if scale is None:
        scale = k.shape[-1]**-0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"
        if head_first:
            raise RuntimeError(
                "Sequences with variable lengths are not supported for head-first mode")
    if head_first:
        q, k, v, block_indices = map(lambda x: rearrange(x, 'b h t d -> b t h d'),
                                     (q, k, v, block_indices))
        g_slc, g_swa = map(lambda x: rearrange(x, 'b h t -> b t h'), (g_slc, g_swa))
        if isinstance(block_counts, torch.Tensor):
            block_counts = rearrange(block_counts, 'b h t -> b t h')

    dtype = q.dtype
    G = q.shape[2] // k.shape[2]
    BS = block_size
    S = block_indices.shape[-1]
    k, v, block_indices = (repeat(x, 'b t h d -> b t (h g) d', g=G) for x in (k, v, block_indices))
    if isinstance(block_counts, torch.Tensor):
        block_counts = repeat(block_counts, 'b t h -> b t (h g)', g=G)
    c = torch.arange(S).repeat_interleave(BS).unsqueeze(1).expand(-1, q.shape[2]).to(q.device)
    q, k, v = map(lambda x: x.float(), (q, k, v))

    o_slc = torch.zeros_like(v)
    o_swa = torch.zeros_like(v) if window_size > 0 else None
    varlen = True
    if cu_seqlens is None:
        varlen = False
        B, T = q.shape[:2]
        cu_seqlens = torch.cat(
            [block_indices.new_tensor(range(0, B * T, T)),
             block_indices.new_tensor([B * T])])

    for i in range(len(cu_seqlens) - 1):
        if not varlen:
            q_b, k_b, v_b, g_slc_b, g_swa_b, i_b = q[i], k[i], v[i], g_slc[i], g_swa[
                i], block_indices[i]
            if isinstance(block_counts, torch.Tensor):
                s_b = block_counts[i]
            else:
                s_b = block_counts
        else:
            T = cu_seqlens[i + 1] - cu_seqlens[i]
            q_b, k_b, v_b, g_slc_b, g_swa_b, i_b = map(
                lambda x: x[0][cu_seqlens[i]:cu_seqlens[i + 1]],
                (q, k, v, g_slc, g_swa, block_indices))
            if isinstance(block_counts, torch.Tensor):
                s_b = block_counts[0][cu_seqlens[i]:cu_seqlens[i + 1]]
            else:
                s_b = block_counts

        i_b = i_b.unsqueeze(-1) * BS + i_b.new_tensor(range(BS))
        # [T, S*BS, HQ]
        i_b = i_b.view(T, block_indices.shape[2], -1).transpose(1, 2)
        for i_q in range(T):
            # [HQ, D]
            q_i = q_b[i_q] * scale
            # [HQ]
            g_slc_i = g_slc_b[i_q]
            # [HQ]
            g_swa_i = g_swa_b[i_q]
            # [S*BS, HQ]
            i_i = i_b[i_q]
            # [HQ]
            if isinstance(block_counts, torch.Tensor):
                s_i = s_b[i_q]
            else:
                s_i = s_b
            # [S*BS, HQ, -1]
            k_i_slc, v_i_slc = map(
                lambda x: x.gather(
                    0,
                    i_i.clamp(0, T - 1).unsqueeze(-1).expand(*i_i.shape, x.shape[-1])), (k_b, v_b))
            # [S*BS, HQ]
            attn_slc = torch.einsum('h d, n h d -> n h', q_i, k_i_slc).masked_fill(
                torch.logical_or(i_i < 0, i_i > i_q) |
                (c >= s_i if block_counts is not None else False), float('-inf'))
            
            # with open('save.txt', 'a+') as f:
            #     # Save tensor with row/column format
            #     f.write("==================================attn_slc data==================================:\n")
            #     # Write each head as a column
            #     for head_idx in range(attn_slc.shape[0]):  # Iterate over HQ heads
            #         # Convert each row's value for this head to string
            #         row_values = [f"{x.item():.6f}" for x in attn_slc[head_idx, :]]
            #         f.write(" ".join(row_values) + "\n")  # Write one line per head
                    
            # print(attn_slc.shape)
            attn_slc = attn_slc.softmax(dim=0)
            # with open('save_sftmx.txt', 'a+') as f:
            #     # Save tensor with row/column format
            #     f.write("==================================attn_slc data==================================:\n")
            #     # Write each head as a column
            #     for head_idx in range(attn_slc.shape[0]):  # Iterate over HQ heads
            #         # Convert each row's value for this head to string
            #         row_values = [f"{x.item():.6f}" for x in attn_slc[head_idx, :]]
            #         f.write(" ".join(row_values) + "\n")  # Write one line per head
            if not varlen:
                o_slc[i, i_q] = torch.einsum('n h, n h v -> h v', attn_slc,
                                             v_i_slc) * g_slc_i.unsqueeze(-1)
            else:
                o_slc[0][cu_seqlens[i] + i_q] = torch.einsum('n h, n h v -> h v', attn_slc,
                                                             v_i_slc) * g_slc_i.unsqueeze(-1)
            if window_size > 0:
                k_i_swa, v_i_swa = map(lambda x: x[max(0, i_q - window_size + 1):i_q + 1],
                                       (k_b, v_b))
                attn_swa = torch.einsum('h d, n h d -> n h', q_i, k_i_swa).softmax(0)
                if not varlen:
                    o_swa[i, i_q] = torch.einsum('n h, n h v -> h v', attn_swa,
                                                 v_i_swa) * g_swa_i.unsqueeze(-1)
                else:
                    o_swa[0][cu_seqlens[i] + i_q] = torch.einsum('n h, n h v -> h v', attn_swa,
                                                                 v_i_swa) * g_swa_i.unsqueeze(-1)

    if head_first:
        o_slc = rearrange(o_slc, 'b t h d -> b h t d')
        o_swa = rearrange(o_swa, 'b t h d -> b h t d')

    return o_slc.to(dtype) + o_swa.to(dtype) if o_swa is not None else o_slc.to(dtype)





if __name__ == "__main__":
    B, T, H, HQ, D, S, block_size, dtype = 2, 64, 1, 16, 32, 2, 32, torch.float16
    # torch.random.manual_seed(84831)
    q = torch.randn((B, T, HQ, D), dtype=dtype, device='cuda').requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device='cuda').requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device='cuda').requires_grad_(True)
    g_slc = torch.ones((B, T, HQ), dtype=dtype, device='cuda').requires_grad_(True)
    g_swa = torch.ones((B, T, HQ), dtype=dtype, device='cuda').requires_grad_(True)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device='cuda')

    block_indices = torch.full((B, T, H, S), T, dtype=torch.long, device='cuda')
    for b in range(B):
        for t in range(T):
            for h in range(H):
                i_i = torch.randperm(max(1, (t // block_size)))[:S]
                # print(i_i)
                block_indices[b, t, h, :len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]
    
    #NOTE: We change first element of block_indices from 0 to others manually, aiming to produce some nan in ref
    block_indices[0][0][0][0] = 4
    block_indices[0][3][0][0] = 2
    block_indices[0][7][0][0] = 10
    block_indices[1][40][0][0] = 2
    block_indices[1][63][0][0] = 7
    block_indices[1][10][0][0] = 5

    block_counts = torch.randint(1, S + 1, (B, T, H), device='cuda')

    ref = naive_nsa(
        q=q,
        k=k,
        v=v,
        g_slc=g_slc,
        g_swa=g_swa,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
    )
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg_slc, g_slc.grad = g_slc.grad.clone(), None

    tri = selection_attention(
        q=q,
        k=k,
        v=v,
        g_slc=g_slc,
        g_swa=g_swa,
        block_indices=block_indices,
        block_size=block_size,
        block_counts=block_counts,
    )
    print(tri)
    print(ref)
    
    #NOTE: We replace nan in ref to 0.0 to match the result of tri and make bwd correct
    ref[torch.isnan(ref)] = 0.0
    
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg_slc, g_slc.grad = g_slc.grad.clone(), None

    # assert_close(" o", ref, tri, 0.004)
    torch.testing.assert_close(ref, tri, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ref_dq, tri_dq, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ref_dk, tri_dk, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ref_dv, tri_dv, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ref_dg_slc, tri_dg_slc, atol=1e-2, rtol=1e-2)