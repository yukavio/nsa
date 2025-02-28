import pytest
import torch
import triton
import triton.language as tl
from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func as flash_attn_func

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
    select_id,
    seq_token_id,
    k_id,
    qk_scale,  #
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    BLOCK_NUM: tl.constexpr,
):

    if STAGE == 2:  # CAUSAL TILE
        lo, hi = k_id, k_id + 1
    else:
        lo, hi = 0, BLOCK_NUM

    lo, hi = lo.to(tl.int32), hi.to(tl.int32)
    start_n = tl.load(select_id + lo) * BLOCK_N
    if STAGE == 1:
        cond = seq_token_id >= start_n + BLOCK_N
    elif STAGE == 2:
        cond = seq_token_id < start_n + BLOCK_N
    else:
        cond = True

    # loop over k, v and update accumulator
    while cond:
        start_n = tl.load(select_id + lo) * BLOCK_N
        start_n = tl.multiple_of(start_n, BLOCK_N)
        cur_k_ptr = tl.advance(K_block_ptr, (0, start_n.to(tl.int32)))
        cur_v_ptr = tl.advance(V_block_ptr, (start_n.to(tl.int32), 0))
        # -- compute qk ----
        k = tl.load(cur_k_ptr)
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
        v = tl.load(cur_v_ptr)
        p = p.to(tl.bfloat16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        lo += 1
        if STAGE == 1:
            cond = (
                False
                if lo >= hi
                else seq_token_id >= tl.load(select_id + lo) * BLOCK_N + BLOCK_N
            )
        elif STAGE == 2:
            cond = False
        else:
            cond = lo < hi

    return acc, l_i, m_i, lo


configs = [
    triton.Config({}, num_stages=s, num_warps=w)
    for s in [3, 4, 5, 6, 7, 8]
    for w in [4, 8, 16, 32]
]


@triton.autotune(configs, key=["HEAD_DIM", "STAGE"])
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    cu_seq_len,
    select_id,
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
    HEAD_DIM: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    NUM_GROUP: tl.constexpr,
    GROUPE_SIZE: tl.constexpr,
    BLOCK_NUM: tl.constexpr,
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
        k_id = 0
        if STAGE & 1:
            acc, l_i, m_i, k_id = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,  #
                select_id + (seq_offset - 1) * BLOCK_NUM,
                token_id - cum_seq_offset,
                k_id,
                qk_scale,  #
                HEAD_DIM,
                BLOCK_N,  #
                4 - STAGE,
                BLOCK_NUM,
            )
        if STAGE & 2:
            acc, l_i, m_i, _ = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,  #
                select_id + (seq_offset - 1) * BLOCK_NUM,
                token_id - cum_seq_offset,
                k_id,
                qk_scale,  #
                HEAD_DIM,
                BLOCK_N,  #
                2,
                BLOCK_NUM,
            )

        # epilogue
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
        m_ptrs = M + (cum_seq_offset + token_id) * GROUPE_SIZE * NUM_GROUP + group_id * GROUPE_SIZE + tl.arange(0, GROUPE_SIZE)
        tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, acc.to(Out.type.element_ty))
