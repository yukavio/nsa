import torch
import triton
import triton.language as tl
from .triton_selection_attn_fwd import _attn_fwd
from .triton_selection_attn_bwd import _attn_bwd, _attn_bwd_preprocess


DEVICE = "cuda"
HAS_FLASH = True
TORCH_HAS_FP8 = False
# B,H,T,D
class _attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q, k, v, cu_seq_len, select_id, causal, sm_scale, block_size, block_num, return_attn_probs=False
    ):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        q_num_head, kv_num_head = q.shape[1], k.shape[1]
        group_size = q_num_head // kv_num_head
        assert group_size >= 16
        o = torch.zeros_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}

        M = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
        num_block = torch.cuda.get_device_properties("cuda").multi_processor_count * 2
        grid = lambda args: (num_block,)
        ctx.grid = grid
        _attn_fwd[grid](
            q,
            k,
            v,
            cu_seq_len,
            select_id,
            sm_scale,
            M,
            o,  #
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            q.shape[0],
            HEAD_DIM=HEAD_DIM_K,  #
            BLOCK_N=block_size,
            STAGE=stage,  #
            NUM_GROUP=kv_num_head,
            GROUPE_SIZE=group_size,
            BLOCK_NUM=block_num,
            **extra_kern_args,
        )

        ctx.save_for_backward(q, k, v, o, M， cu_seq_len, select_id)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        if return_attn_probs:
            return o, M, None
        else:
            return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M, cu_seq_len, select_id = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        bs = cu_seq_len.numel() - 1
        num_token, q_num_head, kv_num_head = q.shape[0], q.shape[1], k.shape[1]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 5
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        # pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        pre_grid = (bs, 32)
        delta = torch.empty_like(M)
        
        _attn_bwd_preprocess[pre_grid](
            o, do, cu_seq_len,  #
            delta,  #
            q_num_head, BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM  #
        )
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q, arg_k, v, ctx.sm_scale, do, dq, dk, dv,  #
            M, delta,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            N_HEAD, N_CTX,  #
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES  #
        )

        return dq, dk, dv, None, None


attention = _attention.apply