import math
import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.nn import init
from einops import rearrange, repeat

def calc_compressed_len(x, stride, size):
    return  (x - size) // stride


def get_autotune_config():
    return [
        triton.Config({'BLOCK_M': bm}, num_warps=nw, num_stages=s) 
        for bm in [16, 32, 64, 128] 
        for nw in [4, 8, 16, 32]
        for s in [2, 3, 4]
    ]

@triton.autotune(
    configs=get_autotune_config(),
    key=['num_heads', 'head_dim', 'block_stride', 'block_size'],
)
@triton.jit
def _compress_fwd(x, w, out, cu_input_len, cu_out_len, num_heads: tl.constexpr,
                  head_dim: tl.constexpr, block_stride: tl.constexpr, block_size: tl.constexpr,
                  BLOCK_M: tl.constexpr):
    bs_id, head_id, start_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq_offset = tl.load(cu_input_len+bs_id)
    seq_upper = tl.load(cu_input_len+bs_id+1)
    out_offset = tl.load(cu_out_len+bs_id)
    out_upper = tl.load(cu_out_len+bs_id+1)
    out_len = out_upper - out_offset
    n_ctx = seq_upper-seq_offset
    
    x_ptr = x+seq_offset*num_heads*head_dim + head_id*head_dim
    out_ptr = out + out_offset*num_heads*head_dim + head_id*head_dim
    
    for task_id in range(start_id, (out_len+BLOCK_M-1)//BLOCK_M, tl.num_programs(2)):
        # task_x_offset = task_id*BLOCK_M
        off_m = tl.arange(0, BLOCK_M) + task_id*BLOCK_M
        off_n = tl.arange(0, head_dim)
        off_k = tl.arange(0, head_dim)
        task_out_ptr = out_ptr + off_m[:, None]*head_dim*num_heads + off_n[None, :]

        accumulator = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)
        for inner_id in range(0, block_size, 1):
            off_acc_m = off_m * block_stride + inner_id
            task_w_ptr = w+inner_id*head_dim*head_dim

            x_data = tl.load(x_ptr+(off_acc_m*num_heads*head_dim)[:, None]+off_k[None,:], mask=off_acc_m[:,None]<n_ctx, other=0) # (BLOCK_M, head_dim)
            w_data = tl.load(task_w_ptr+off_k[:, None]*head_dim+off_n[None,:]) # head_dim, head_dim
            accumulator += tl.dot(x_data, w_data)
        accumulator = accumulator.to(tl.bfloat16)
        c_mask = (off_m[:, None]<out_len)
        tl.store(task_out_ptr, accumulator, mask=c_mask)




@triton.autotune(
    configs=get_autotune_config(),
    key=['num_heads', 'head_dim', 'block_stride', 'block_size'],
)
@triton.jit
def _compress_bwd_dw(
    x, grad_out, grad_w,
    cu_input_len, cu_out_len,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_stride: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_M: tl.constexpr
):
    bs_id, head_id, j = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    seq_offset = tl.load(cu_input_len + bs_id)
    seq_upper = tl.load(cu_input_len + bs_id + 1)
    out_offset = tl.load(cu_out_len + bs_id)
    out_upper = tl.load(cu_out_len + bs_id + 1)
    n_ctx = seq_upper - seq_offset
    out_len = out_upper - out_offset
    
    x_ptr = x + seq_offset * num_heads * head_dim + head_id * head_dim
    grad_out_ptr = grad_out + out_offset * num_heads * head_dim + head_id * head_dim
    grad_w_ptr = grad_w + j * head_dim * head_dim
    
    accumulator = tl.zeros((head_dim, head_dim), dtype=tl.float32)
    for i in range((out_len + BLOCK_M - 1) // BLOCK_M):
        off_i = i * BLOCK_M + tl.arange(0, BLOCK_M)
        valid_i = off_i < out_len
        
        input_idx = off_i * block_stride + j
        valid_input = input_idx < n_ctx
        
        # [BLOCK_M, head_dim]
        x_data = tl.load(
            x_ptr + (input_idx * num_heads * head_dim)[:, None] + tl.arange(0, head_dim)[None, :],
            mask=valid_input[:, None] & valid_i[:, None],
            other=0.0
        )
        
        # [BLOCK_M, head_dim]
        grad_data = tl.load(
            grad_out_ptr + (off_i * num_heads * head_dim)[:, None] + tl.arange(0, head_dim)[None, :],
            mask=valid_i[:, None],
            other=0.0
        )
        
        accumulator += tl.dot(x_data.T, grad_data)
    
    off_m = tl.arange(0, head_dim)[:, None]
    off_n = tl.arange(0, head_dim)[None, :]
    grad_w_ptr = grad_w_ptr + off_m * head_dim + off_n
    tl.atomic_add(grad_w_ptr, accumulator.to(tl.float32))
        
        
        
@triton.autotune(
    configs=get_autotune_config(),
    key=['num_heads', 'head_dim', 'block_stride', 'block_size'],
)        
@triton.jit
def _compress_bwd_dx(
    grad_out, w, grad_x,
    cu_input_len, cu_out_len,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_stride: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_M: tl.constexpr
):
    bs_id, head_id, start_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq_offset = tl.load(cu_input_len + bs_id)
    seq_upper = tl.load(cu_input_len + bs_id + 1)
    out_offset = tl.load(cu_out_len + bs_id)
    out_upper = tl.load(cu_out_len + bs_id + 1)
    n_ctx = seq_upper - seq_offset
    out_len = out_upper - out_offset


    grad_out_ptr = grad_out + out_offset * num_heads * head_dim + head_id * head_dim
    grad_x_ptr = grad_x + seq_offset * num_heads * head_dim + head_id * head_dim
    w_ptr = w
    
    total_tasks = (out_len + BLOCK_M - 1) // BLOCK_M
    for task_id in range(start_id * 4, total_tasks, tl.num_programs(2) * 4):
        off_m = tl.arange(0, BLOCK_M) + task_id * BLOCK_M
        off_n = tl.arange(0, head_dim)
        off_k = tl.arange(0, head_dim)
        
        grad_out_data = tl.load(
            grad_out_ptr + off_m[:, None] * num_heads * head_dim + off_n[None, :],
            mask=off_m[:, None] < out_len,
            other=0.0
        )
        
        for j in range(block_size):
            w_ptr_j = w_ptr + j * head_dim * head_dim
            w_data = tl.load(w_ptr_j + off_k[:, None] * head_dim + off_n[None, :])
            
            input_idx = off_m * block_stride + j
            valid_input = input_idx < n_ctx
            
            grad_x_ptr_j = grad_x_ptr + (input_idx * num_heads * head_dim)[:, None] + off_k[None, :]
            
            accumulator_j = tl.dot(grad_out_data, tl.trans(w_data))
            accumulator_j = accumulator_j.to(tl.float32)
            
            tl.atomic_add(grad_x_ptr_j, accumulator_j, mask=valid_input[:, None])
        
        
        
        
# k/v: [num_token, NUM_HEAD, HEAD_DIM]
# w: [block_size*HEAD_DIM, HEAD_DIM]
class _compress_kv(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, k, v, w_k, w_v, cu_seq_len, block_stride, block_size
    ):
        NUM_HEAD, HEAD_DIM = k.shape[1:]
        cu_seq_len_cpu = cu_seq_len.tolist()
        pre = 0
        cu_out_len = [0]
        for x in cu_seq_len_cpu[1:]:
            cu_out_len.append(cu_out_len[-1] + calc_compressed_len(x-pre, block_stride, block_size))
            pre = x
        out_len = cu_out_len[-1]
        dtype = torch.bfloat16
        compressed_k = torch.empty(out_len, NUM_HEAD, HEAD_DIM, dtype=dtype, device=k.device)
        compressed_v = torch.empty(out_len, NUM_HEAD, HEAD_DIM, dtype=dtype, device=k.device)
        cu_out_len = torch.tensor(cu_out_len, device=cu_seq_len.device, dtype=torch.int32)
        
        grid = lambda args: (cu_seq_len.numel()-1, NUM_HEAD, 128)
        ctx.grid = grid
        _compress_fwd[grid](
            k, w_k, compressed_k, cu_seq_len, cu_out_len, NUM_HEAD, HEAD_DIM, block_stride, block_size
        )
        _compress_fwd[grid](
            v, w_v, compressed_v, cu_seq_len, cu_out_len, NUM_HEAD, HEAD_DIM, block_stride, block_size
        )

        num_head_tensor = torch.tensor(NUM_HEAD, device=k.device)
        head_dim_tensor = torch.tensor(HEAD_DIM, device=k.device)
        block_stride_tensor = torch.tensor(block_stride, device=k.device)
        block_size_tensor = torch.tensor(block_size, device=k.device)
        
        ctx.save_for_backward(
            k, v, w_k, w_v, 
            cu_seq_len, cu_out_len,
            num_head_tensor, head_dim_tensor,
            block_stride_tensor, block_size_tensor
        )
        
        return compressed_k, compressed_v, cu_out_len
    

    @staticmethod
    def backward(ctx, dck, dcv, _):
        k, v, w_k, w_v, cu_seq_len, cu_out_len, num_head_tensor, head_dim_tensor, block_stride_tensor, block_size_tensor = ctx.saved_tensors
        
        NUM_HEAD = num_head_tensor.item()
        HEAD_DIM = head_dim_tensor.item()
        block_stride = block_stride_tensor.item()
        block_size = block_size_tensor.item()
        
        dw_k = torch.zeros_like(w_k, dtype=torch.float32)
        dw_v = torch.zeros_like(w_v, dtype=torch.float32)
        
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        
        grid = lambda meta: (cu_seq_len.numel() - 1, NUM_HEAD, block_size)
        

        _compress_bwd_dx[grid](
            dck, w_k, dk, 
            cu_seq_len, cu_out_len,
            NUM_HEAD, HEAD_DIM,
            block_stride, block_size, 
        )
        
        _compress_bwd_dx[grid](
            dcv, w_v, dv, 
            cu_seq_len, cu_out_len, 
            NUM_HEAD, HEAD_DIM,
            block_stride, block_size, 
        )
        
        _compress_bwd_dw[grid](
            k, dck, dw_k,
            cu_seq_len, cu_out_len,
            NUM_HEAD, HEAD_DIM,
            block_stride, block_size,
        )
        
        _compress_bwd_dw[grid](
            v, dcv, dw_v,
            cu_seq_len, cu_out_len,
            NUM_HEAD, HEAD_DIM,
            block_stride, block_size,
        )
        
        return dk, dv, dw_k, dw_v, None, None, None
    

compress_kv = _compress_kv.apply


class KVCompressorVarlen(nn.Module):
    def __init__(self, block_stride, block_size, head_dim, device, dtype):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.block_stride = block_stride
        self.block_size = block_size
        self.k_weight = nn.Parameter(torch.empty(block_size*head_dim, head_dim, **factory_kwargs))
        self.v_weight = nn.Parameter(torch.empty(block_size*head_dim, head_dim, **factory_kwargs))
        self.reset_parameters()

    def forward(self, k, v, cu_seq_len, group=None):
        compress_k, compress_v, cu_out_len = compress_kv(k, v, self.k_weight,
            self.v_weight, cu_seq_len, self.block_stride, self.block_size)
        return compress_k, compress_v, cu_out_len
    
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.k_weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.v_weight, a=math.sqrt(5))


class KVCompressor(nn.Module):
    def __init__(self, block_stride, block_size, head_dim, device, dtype):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.block_stride = block_stride
        self.block_size = block_size
        self.compressor_k = nn.Conv1d(head_dim, head_dim, block_size, block_stride, bias=False, **factory_kwargs)
        self.compressor_v = nn.Conv1d(head_dim, head_dim, block_size, block_stride, bias=False, **factory_kwargs)

    def forward(self, k, v, cu_seq_len, group=None):
        BT, H, D = k.shape
        B = cu_seq_len.shape[0] - 1
        ck = k.reshape(B, -1, H, D).permute(0, 2, 3, 1).reshape(B*H, D, -1)  # B, T, H, D -> BH, D, T
        cv = v.reshape(B, -1, H, D).permute(0, 2, 3, 1).reshape(B*H, D, -1)
        ck = self.compressor_k(ck).reshape(B, H, D, -1).permute(0, 3, 1, 2)
        cv = self.compressor_v(cv).reshape(B, H, D, -1).permute(0, 3, 1, 2)
        if group is not None:
            ck = repeat(ck, "b s h d -> b s (h g) d", g=group).contiguous()
            cv = repeat(cv, "b s h d -> b s (h g) d", g=group).contiguous()

        return ck.contiguous(), cv.contiguous(), cu_seq_len

    
