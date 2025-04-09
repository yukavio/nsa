# This File is copy and edited from https://triton-lang.org/main/getting-started/tutorials/index.html
import torch
import math
import triton
import triton.language as tl
from .utils import softmax_kernel

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    block_stride: tl.constexpr, block_size: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    lo = 0
    if STAGE==3:
        hi = (((start_m*BLOCK_M + BLOCK_M)-block_size+block_stride-1)//block_stride+BLOCK_N-1)//BLOCK_N*BLOCK_N
    else:
        hi = N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        qk = tl.dot(q, k)
        if STAGE == 3:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])*block_stride+block_size
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
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
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

        p = p.to(q.dtype)
        acc += tl.dot(p, v)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    
    return acc, l_i, m_i



# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64]\
    for BN in [64]\
    for s in [3, 4, 7]\
    for w in [4, 8]\
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


# @triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H, N_CTX, Q_CTX, #
              block_stride: tl.constexpr,
              block_size: tl.constexpr,
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_z = tl.program_id(1).to(tl.int64)
    off_h = tl.program_id(2).to(tl.int64)
    qo_offset = off_z * stride_qz + off_h * stride_qh
    kv_offset = off_z * stride_kz + off_h * stride_kh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qo_offset,
        shape=(Q_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qo_offset,
        shape=(Q_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    # stage 2: on-band

    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                    start_m, qk_scale,  #
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                    block_stride, block_size, #
                                    STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                    )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_z * Q_CTX * H + offs_m * H + off_h
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0,))
    if start_m == 0 and STAGE==3:
        clear_block_ptr = tl.make_block_ptr(
            base=Out + qo_offset,
            shape=(block_size, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(0, 0),
            block_shape=(block_size, HEAD_DIM),
            order=(1, 0),
        )

        tl.store(clear_block_ptr, tl.zeros([block_size, HEAD_DIM], dtype=Out.type.element_ty))


@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         Delta,  #
                         Z, H, N_CTX,  #
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  #
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_z = tl.program_id(1)
    off_h = tl.program_id(2)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + off_z * HEAD_DIM * N_CTX * H + off_h * HEAD_DIM + off_m[:, None] * HEAD_DIM * H + off_n[None, :])
    do = tl.load(DO + off_z * HEAD_DIM * N_CTX * H + off_h * HEAD_DIM + off_m[:, None] * HEAD_DIM * H + off_n[None, :])
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_z * N_CTX * H + off_h + off_m * H, delta)


@triton.jit
def bwd_attn_mul(attn_ptr, d_attn_ptr, o_ptr, H:tl.constexpr, CTX_Q:tl.constexpr, CTX_KV: tl.constexpr, CTX_KV_UPPER: tl.constexpr):
    off = tl.program_id(0).to(tl.int64) * H * CTX_Q * CTX_KV + tl.program_id(1).to(tl.int64) * CTX_Q * CTX_KV + tl.program_id(2).to(tl.int64) * CTX_KV
    attn_ptr += off
    d_attn_ptr += off
    o_ptr += off
    off_d = tl.arange(0, CTX_KV_UPPER)
    # load
    attn = tl.load(attn_ptr+off_d, mask=off_d<CTX_KV, other=0)
    d_attn = tl.load(d_attn_ptr+off_d, mask=off_d<CTX_KV, other=0)
    t = attn*d_attn
    sum_data = tl.sum(t, axis=0)
    data = t-sum_data*attn
    # write-back
    tl.store(o_ptr+off_d, data, mask=off_d<CTX_KV)


@triton.jit
def _attn_bwd_only_dkv(Q, K, V, sm_scale,  #
              DO,  #
              DQ, DK, DV,  #
              M, D,
              # shared by Q/K/V/DO.
              stride_z, stride_h, stride_tok, stride_d,  #
              stride_kz, stride_kh, stride_ktok, stride_kd, #
              H, Q_CTX, KV_CTX,  #
              block_stride: tl.constexpr,
              block_size: tl.constexpr,
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,
              CAUSAL: tl.constexpr):

    bid = tl.program_id(1)
    hid = tl.program_id(2)
    off_chz = (bid * Q_CTX*H+hid).to(tl.int64)
    adj_q = (stride_h * hid + stride_z * bid).to(tl.int64)
    adj_kv = (stride_kh * hid + stride_kz * bid).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj_q
    K += adj_kv
    V += adj_kv
    DO += adj_q
    DQ += adj_q
    DK += adj_kv
    DV += adj_kv
    M += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_ktok + offs_k[None, :] * stride_kd, mask=offs_n[:, None]<KV_CTX, other=0)
    v = tl.load(V + offs_n[:, None] * stride_ktok + offs_k[None, :] * stride_kd, mask=offs_n[:, None]<KV_CTX, other=0)

    #num_steps = Q_CTX  // BLOCK_M1
    if CAUSAL:
        start_m = (start_n * block_stride + block_size) // BLOCK_M1 * BLOCK_M1
        num_steps = (Q_CTX-start_m)  // BLOCK_M1
    else:
        start_m = 0
        num_steps = Q_CTX  // BLOCK_M1

    # Compute dK and dV for non-masked blocks.
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m*H)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if CAUSAL:
            mask = (offs_m[None, :] >= offs_n[:, None]*block_stride+block_size)
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs).to(dk.dtype)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(dk.dtype)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m*H)
        # Compute dP and dS.
        dpT = tl.dot(v.to(do.dtype), tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(dk.dtype)
        dk += tl.dot(dsT.to(qT.dtype), tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok

    dv_ptrs = DV + offs_n[:, None] * stride_ktok + offs_k[None, :] * stride_kd
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_ktok + offs_k[None, :] * stride_kd
    tl.store(dk_ptrs, dk)


@triton.jit
def _attn_bwd_only_dq(Q, K, V, sm_scale,  #
              DO,  #
              DQ, DK, DV,  #
              M, D,
              # shared by Q/K/V/DO.
              q_stride_z, q_stride_h, q_stride_tok, q_stride_d,  #
              k_stride_z, k_stride_h, k_stride_tok, k_stride_d,  #
              H, Q_CTX, KV_CTX,  #
              block_stride: tl.constexpr,
              block_size: tl.constexpr,
              BLOCK_M2: tl.constexpr,  #
              BLOCK_N2: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,
              CAUSAL: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bid = tl.program_id(1)
    hid = tl.program_id(2)
    off_chz = (bid * Q_CTX*H+hid).to(tl.int64)
    adj_q = (q_stride_h * hid + q_stride_z * bid).to(tl.int64)
    adj_kv = (k_stride_h * hid + k_stride_z * bid).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj_q
    K += adj_kv
    V += adj_kv
    DO += adj_q
    DQ += adj_q
    DK += adj_kv
    DV += adj_kv
    M += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_m = 0

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    if CAUSAL:
        end_n = (((start_m + BLOCK_M2)-block_size+block_stride-1)//block_stride+BLOCK_N2-1)//BLOCK_N2*BLOCK_N2
    else:
        end_n = KV_CTX

    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * q_stride_tok + offs_k[None, :] * q_stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * q_stride_tok + offs_k[None, :] * q_stride_d)

    m = tl.load(M + offs_m*H)
    m = m[:, None]

    # stage 2
    num_steps = end_n // BLOCK_N2
    start_n = 0
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * k_stride_tok + offs_k[:, None] * k_stride_d
    vT_ptrs = V + offs_n[None, :] * k_stride_tok + offs_k[:, None] * k_stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m*H)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs, mask=offs_n[None, :]<KV_CTX, other=0)
        vT = tl.load(vT_ptrs, mask=offs_n[None, :]<KV_CTX, other=0)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if CAUSAL:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :]*block_stride+block_size)
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(q.dtype)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * k_stride_tok
        vT_ptrs += step_n * k_stride_tok

    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * q_stride_tok + offs_k[None, :] * q_stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)
    

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, block_stride, block_size, causal, sm_scale, pool_num_kv_head=0, pool_bs=0, pool_kernel_size=0, pool_stride=0, pool_padding=0, select_block_count=0, fused=False):
        # B, T, H, D

        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        if sm_scale == None:
            sm_scale = 1 / math.sqrt(HEAD_DIM_Q)

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        grid = lambda args: (triton.cdiv(q.shape[1], args["BLOCK_M"]), q.shape[0], q.shape[2])
        ctx.grid = grid
        _attn_fwd[grid](
            q, k, v, sm_scale, M, o,  #
            q.stride(0), q.stride(2), q.stride(1), q.stride(3),  #
            k.stride(0), k.stride(2), k.stride(1), k.stride(3),  #
            v.stride(0), v.stride(2), v.stride(1), v.stride(3),  #
            o.stride(0), o.stride(2), o.stride(1), o.stride(3),  #
            q.shape[0], q.shape[2],  #
            N_CTX=k.shape[1], Q_CTX=q.shape[1],  #
            block_stride=block_stride,
            block_size=block_size,
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,  #
            BLOCK_M=64,
            BLOCK_N=64,
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        ctx.block_stride = block_stride
        ctx.block_size = block_size
        s = torch.einsum("bthd, bshd->bhts", q, k)
        #s = torch.nn.functional.softmax(s, dim=-1)
        softmax_grid = (256, )
        n_row, n_col, block_size = s.numel()//s.shape[-1], s.shape[-1], triton.next_power_of_2(s.shape[-1])
        softmax_kernel[softmax_grid](s, s, s.stride(2), s.stride(2), n_row, n_col, block_size, 4)
        
        if fused:
            s = s.reshape(pool_bs, pool_num_kv_head, -1, *s.shape[-2:]).sum(2)
            s = s.reshape(-1, *s.shape[2:])
            s = torch.nn.functional.avg_pool1d(s, pool_kernel_size, pool_stride, pool_padding, True)
            s = s.reshape(pool_bs, pool_num_kv_head, *s.shape[-2:])  # -> B, H, T1, T2
            indices = torch.topk(s, select_block_count, dim=3).indices # B, H, T1, S
            indices = indices.transpose(1, 2)
        
            return o, indices
        else:
            return o, s

    @staticmethod
    def backward(ctx, do, ds):
        print(f"ds sum: {ds.sum()}")
        q, k, v, o, M = ctx.saved_tensors
        q = q.contiguous()
        o = o.contiguous()
        do = do.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        assert do.is_contiguous()
        assert q.stride() == o.stride() == do.stride()
        assert k.stride() == v.stride()
        dq = torch.empty_like(q)
        dk = torch.zeros_like(k)
        dv = torch.empty_like(v)
        BATCH, Q_CTX, Q_HEAD = q.shape[:3]
        _, KV_CTX, KV_HEAD = k.shape[:3]
        NUM_WARPS, NUM_STAGES = 4, 5
        
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        assert Q_CTX % PRE_BLOCK == 0
        pre_grid = (Q_CTX // PRE_BLOCK, BATCH, Q_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o, do,  #
            delta,  #
            BATCH, Q_HEAD, Q_CTX,  #
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM  #
        )

        BLOCK_M_kv, BLOCK_N_kv = 64, 64
        grid_kv = (triton.cdiv(KV_CTX, BLOCK_N_kv), BATCH, KV_HEAD)
        _attn_bwd_only_dkv[grid_kv](
            q, arg_k, v, ctx.sm_scale, do, dq, dk, dv,  #
            M, delta,  #
            q.stride(0), q.stride(2), q.stride(1), q.stride(3),  #
            k.stride(0), k.stride(2), k.stride(1), k.stride(3), #
            Q_HEAD, Q_CTX, KV_CTX, #
            ctx.block_stride,
            ctx.block_size,
            BLOCK_M1=BLOCK_M_kv, BLOCK_N1=BLOCK_N_kv,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            CAUSAL=ctx.causal,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES  #
        )

        BLOCK_M_q, BLOCK_N_q = 128, 32
        grid_q = (Q_CTX // BLOCK_M_q, BATCH, Q_HEAD)

        _attn_bwd_only_dq[grid_q](
            q, arg_k, v, ctx.sm_scale, do, dq, dk, dv,  #
            M, delta,  #
            q.stride(0), q.stride(2), q.stride(1), q.stride(3),  #
            k.stride(0), k.stride(2), k.stride(1), k.stride(3),  #
            Q_HEAD, Q_CTX, KV_CTX, #
            ctx.block_stride, ctx.block_size, #
            BLOCK_M2=BLOCK_M_q, BLOCK_N2=BLOCK_N_q,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            CAUSAL=ctx.causal,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES  #
        )
        # if ds is not None:
        #     # Recompute s
        #     s = torch.einsum("bthd, bshd->bhts", q, k) 
        #     #print(torch.cuda.max_memory_allocated()/1024**3)
        #     softmax_grid = (256, )
        #     n_row, n_col, block_size = s.numel()//s.shape[-1], s.shape[-1], triton.next_power_of_2(s.shape[-1])
        #     softmax_kernel[softmax_grid](s, s, s.stride(2), s.stride(2), n_row, n_col, block_size, 4)
        #     #s = torch.nn.functional.softmax(s, dim=-1)
        #     #print(torch.cuda.max_memory_allocated()/1024**3)
            
        #     # backward softmax
        #     #d_attn = torch.empty_like(s)
        #     d_attn = s
        #     grid = (s.shape[0], s.shape[1], s.shape[2])
        #     bwd_attn_mul[grid](s, ds, d_attn, s.shape[1], s.shape[2], s.shape[3], triton.next_power_of_2(s.shape[3]))
        #     dq += torch.einsum("bhts,bshd->bthd", d_attn, k.to(d_attn.dtype))
        #     dk += torch.einsum("bhts,bthd->bshd", d_attn, q.to(d_attn.dtype))
        # print(torch.cuda.max_memory_allocated()/1024**3)
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None


flash_attn_func = _attention.apply
