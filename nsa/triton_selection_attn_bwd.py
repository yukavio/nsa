import torch
import triton
import triton.language as tl


@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         cu_seq_len,
                         Delta,  #
                         q_num_head: tl.constexpr,
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  #
                         ):
    seq_id = tl.program_id(0)
    token_offset = tl.load(cu_seq_len + seq_id)
    token_upper = tl.load(cu_seq_len + seq_id + 1)
    N_CTX = token_upper - token_offset
    
    for start_t in tl.range(tl.program_id(1), N_CTX, tl.num_programs(1)):
        off_t = token_offset + start_t * BLOCK_M + tl.arange(0, BLOCK_M)
        off_h = tl.arange(0, HEAD_DIM)
        # load
        o = tl.load(O + + off_m[:, None] * HEAD_DIM + off_n[None, :])
        do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
        delta = tl.sum(o * do, axis=1)
        # write-back
        tl.store(Delta + off_hz * N_CTX + off_m, delta)
    
    