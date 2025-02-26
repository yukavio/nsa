import pytest
import torch
import triton
import triton.language as tl
import triton.tools.experimental_descriptor
from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func as flash_attn_func

# ENABLE_LHS_TO_TMEM is an experimental environment variable for Blackwell.
# If it is set to 1 it can improve performance of Blackwell attention. However,
# it defaults to 0 as it is known to cause correctness issues outside of the
# _attn_fwd_tma kernel below.

DEVICE = "cuda"
HAS_FLASH = True
TORCH_HAS_FP8 = False


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,  #
    K_block_ptr,
    V_block_ptr,  #
    seq_token_id,
    qk_scale,  #
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    N_CTX: tl.constexpr,
):

    if STAGE == 1:
        # lo, hi = 0, ((((seq_token_id + BLOCK_N - 1) // BLOCK_N)) - 1) * BLOCK_N
        lo, hi = 0, (seq_token_id // BLOCK_N) * BLOCK_N - 31
        # tl.device_print("", hi)
    elif STAGE == 2:
        # tl.device_print("", (((seq_token_id + BLOCK_N - 1) // BLOCK_N) - 1) * BLOCK_N)
        lo = (seq_token_id // BLOCK_N) * BLOCK_N
        hi = lo + BLOCK_N
        lo = tl.multiple_of(lo, BLOCK_N)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    lo, hi = lo.to(tl.int32), hi.to(tl.int32)
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = seq_token_id >= start_n + tl.arange(0, BLOCK_N)[None, :]
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e10)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        p = p.to(tl.bfloat16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


configs = [
    triton.Config({"BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BN in [128, 256]
    for s in [3, 4, 7]
    for w in [4, 8, 16, 32]
]


@triton.autotune(configs, key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    cu_seq_len,
    sm_scale,
    M,
    Out,  #
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    num_token,
    num_seq,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    NUM_GROUP: tl.constexpr,
    GROUPE_SIZE: tl.constexpr,
):

    start_id = tl.program_id(0)
    seq_offset = 1
    cum_seq_offset = 0
    cum_seq_upper = tl.load(cu_seq_len + seq_offset)
    N_CTX = cum_seq_upper

    for task_id in tl.range(start_id, num_token * NUM_GROUP, tl.num_programs(0)):
        token_id = task_id // NUM_GROUP
        group_id = task_id % NUM_GROUP

        if task_id >= cum_seq_upper * NUM_GROUP:
            seq_offset += 1
            cum_seq_offset = cum_seq_upper
            cum_seq_upper = tl.load(cu_seq_len + seq_offset)
            N_CTX = cum_seq_upper - cum_seq_offset

        q_offset = (
            token_id.to(tl.int64) * stride_qt
            + group_id.to(tl.int64) * stride_qh * GROUPE_SIZE
        )

        kv_offset = (
            cum_seq_offset.to(tl.int64) * stride_kt + group_id.to(tl.int64) * stride_kh
        )

        # block pointers
        Q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset,
            shape=(GROUPE_SIZE, HEAD_DIM),
            strides=(stride_qh, 1),
            offsets=(0, 0),
            block_shape=(GROUPE_SIZE, HEAD_DIM),
            order=(1, 0),
        )

        V_block_ptr = tl.make_block_ptr(
            base=V + kv_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vt, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + kv_offset,
            shape=(HEAD_DIM, N_CTX),
            strides=(1, stride_kt),
            offsets=(0, 0),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(0, 1),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + q_offset,
            shape=(GROUPE_SIZE, HEAD_DIM),
            strides=(stride_qh, 1),
            offsets=(0, 0),
            block_shape=(GROUPE_SIZE, HEAD_DIM),
            order=(1, 0),
        )
        # initialize offsets
        # offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        # initialize pointer to m and l
        m_i = tl.zeros([GROUPE_SIZE], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([GROUPE_SIZE], dtype=tl.float32) + 1.0
        acc = tl.zeros([GROUPE_SIZE, HEAD_DIM], dtype=tl.float32)
        # load scales
        qk_scale = sm_scale
        qk_scale *= 1.44269504  # 1/log(2)
        # load q: it will stay in SRAM throughout
        q = tl.load(Q_block_ptr)
        # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
        # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
        if STAGE & 1:
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,  #
                token_id - cum_seq_offset,
                qk_scale,  #
                HEAD_DIM,
                BLOCK_N,  #
                4 - STAGE,
                N_CTX,
            )
        if STAGE & 2:
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,  #
                token_id - cum_seq_offset,
                qk_scale,  #
                HEAD_DIM,
                BLOCK_N,  #
                2,
                N_CTX,
            )

        # epilogue
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
        # m_ptrs = M + off_hz * N_CTX + offs_m
        # tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, acc.to(Out.type.element_ty))


# B,H,T,D
class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seq_len, causal, sm_scale):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        q_num_head, kv_num_head = q.shape[1], k.shape[1]
        group_size = q_num_head // kv_num_head
        assert group_size >= 16
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}

        M = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
        num_block = torch.cuda.get_device_properties("cuda").multi_processor_count
        grid = lambda args: (num_block,)
        ctx.grid = grid
        _attn_fwd[grid](
            q,
            k,
            v,
            cu_seq_len,
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
            cu_seq_len.shape[0] - 1,
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,  #
            NUM_GROUP=kv_num_head,
            GROUPE_SIZE=group_size,
            **extra_kern_args,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o


attention = _attention.apply
