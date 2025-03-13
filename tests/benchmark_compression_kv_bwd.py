import torch
from nsa.compression_kv import compress_kv, calc_compressed_len, _compress_bwd_dw, _compress_bwd_dx, _compress_fwd
import triton


num_warm_up = 5

bs, seqlen, head_dim, kv_num_head = 16, 1024 * 64, 128, 4
block_size, block_stride = 64, 16
dtype = torch.bfloat16
device = "cuda"
torch.set_default_device(device)
torch.manual_seed(3)

k = torch.randn(bs * seqlen, kv_num_head, head_dim, dtype=dtype, requires_grad=True)
v = torch.randn(bs * seqlen, kv_num_head, head_dim, dtype=dtype, requires_grad=True)
w_k = torch.randn(block_size*head_dim, head_dim, dtype=dtype, requires_grad=True)
w_v = torch.randn(block_size*head_dim, head_dim, dtype=dtype, requires_grad=True)
seq_len = torch.Tensor([0] + [seqlen] * bs)

cu_seq_len = torch.cumsum(seq_len, dim=0).to(torch.int32).to(device)

c_k, c_v =  compress_kv(k, v, w_k, w_v, cu_seq_len, block_stride, block_size)

target = torch.randn_like(c_k)

# test tflops 
def calculate_flops(BATCH_SIZE, SEQ_LENGTH, HEAD_DIM, KV_NUM_HEADS, BLOCK_SIZE, BLOCK_STRIDE):
    compressed_len = max(0, (SEQ_LENGTH - BLOCK_SIZE + BLOCK_STRIDE) // BLOCK_STRIDE)
    total_out_tokens = BATCH_SIZE * compressed_len
    
    single_fwd_flops = total_out_tokens * KV_NUM_HEADS * (2 * BLOCK_SIZE * HEAD_DIM * HEAD_DIM)
    fwd_flops = single_fwd_flops
    
    dw_flops = total_out_tokens * KV_NUM_HEADS * (2 * BLOCK_SIZE * HEAD_DIM * HEAD_DIM)
    dx_flops = total_out_tokens * KV_NUM_HEADS * (2 * BLOCK_SIZE * HEAD_DIM * HEAD_DIM)
    
    
    return fwd_flops, dw_flops, dx_flops

# 使用测试参数计算
params = {
    "BATCH_SIZE": bs,
    "SEQ_LENGTH": seqlen,
    "HEAD_DIM": head_dim,
    "KV_NUM_HEADS": kv_num_head,
    "BLOCK_SIZE": block_size,
    "BLOCK_STRIDE": block_stride
}

fwd_flops, dw_flops, dx_flops = calculate_flops(**params)

# warm up
print("==========================Benchmark forward start==========================")
def init_fwd_data():
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
    return grid, compressed_k, compressed_v, cu_out_len, NUM_HEAD, HEAD_DIM

grid, compressed_k, compressed_v, cu_out_len, NUM_HEAD, HEAD_DIM = init_fwd_data()
def test_forward():
    _compress_fwd[grid](
        k, w_k, compressed_k, cu_seq_len, cu_out_len, NUM_HEAD, HEAD_DIM, block_stride, block_size
    )

for _ in range(num_warm_up):
    test_forward()
    
perf = lambda ms: fwd_flops * 1e-12 / (ms * 1e-3)

ms_forward_triton = triton.testing.do_bench(
    lambda: test_forward()
)

print(f"Triton Forward: {perf(ms_forward_triton):.2f} TFLOPs | Time: {ms_forward_triton:.2f}ms")

print("==========================Benchmark forward end==========================")

torch.cuda.synchronize()


print("==========================Benchmark backward start==========================")

total_bwd_flops = dw_flops + dx_flops
def init_bwd_data():
    NUM_HEAD, HEAD_DIM = k.shape[1:]
    cu_seq_len_cpu = cu_seq_len.tolist()
    pre = 0
    cu_out_len = [0]
    for x in cu_seq_len_cpu[1:]:
        cu_out_len.append(cu_out_len[-1] + calc_compressed_len(x-pre, block_stride, block_size))
        pre = x
    out_len = cu_out_len[-1]
    dtype = torch.bfloat16
    cu_out_len = torch.tensor(cu_out_len, device=cu_seq_len.device, dtype=torch.int32)     
    dw_k = torch.zeros_like(w_k, dtype=torch.float32)
    dw_v = torch.zeros_like(w_v, dtype=torch.float32)

    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)

    dw_k = torch.zeros_like(w_k, dtype=torch.float32)
    dw_v = torch.zeros_like(w_v, dtype=torch.float32)
        
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
        
    dck = torch.zeros(out_len, NUM_HEAD, HEAD_DIM, dtype=dtype, device=k.device)
    dcv = torch.zeros(out_len, NUM_HEAD, HEAD_DIM, dtype=dtype, device=k.device)
    return dw_k, dw_v, dk, dv, dck, dcv, cu_seq_len, cu_out_len, NUM_HEAD, HEAD_DIM, block_stride, block_size

    

dw_k, dw_v, dk, dv, dck, dcv, cu_seq_len, cu_out_len, NUM_HEAD, HEAD_DIM, block_stride, block_size = init_bwd_data()

def dw_backward():
    grid = lambda meta: (cu_seq_len.numel()-1, NUM_HEAD, block_size)
    _compress_bwd_dw[grid](
        k, dck, dw_k,
        cu_seq_len, cu_out_len,
        NUM_HEAD, HEAD_DIM,
        block_stride, block_size,
        # BLOCK_M = 64
    )
    

def dx_backward():
    grid = lambda meta: (cu_seq_len.numel()-1, NUM_HEAD, block_size)
    _compress_bwd_dx[grid](
        dck, w_k, dk, 
        cu_seq_len, cu_out_len,
        NUM_HEAD, HEAD_DIM,
        block_stride, block_size, 
        # BLOCK_M = 16
    )
    
def full_backward():
    dw_backward()
    dx_backward()
# warm up
for _ in range(num_warm_up):
    full_backward()
    dw_backward()
    dx_backward()
dw_k, dw_v, dk, dv, dck, dcv, cu_seq_len, cu_out_len, NUM_HEAD, HEAD_DIM, block_stride, block_size = init_bwd_data()
perf_dw = lambda ms: dw_flops * 1e-12 / (ms * 1e-3)
dw_k, dw_v, dk, dv, dck, dcv, cu_seq_len, cu_out_len, NUM_HEAD, HEAD_DIM, block_stride, block_size = init_bwd_data()
perf_dx = lambda ms: dx_flops * 1e-12 / (ms * 1e-3)
dw_k, dw_v, dk, dv, dck, dcv, cu_seq_len, cu_out_len, NUM_HEAD, HEAD_DIM, block_stride, block_size = init_bwd_data()
perf_total = lambda ms: (dw_flops + dx_flops) * 1e-12 / (ms * 1e-3)

torch.cuda.synchronize()

ms_dx = triton.testing.do_bench(dx_backward)
torch.cuda.synchronize()
print(f"Triton Backward dx only: {perf_dx(ms_dx):.2f} TFLOPs | Time: {ms_dx:.2f}ms")

ms_dw = triton.testing.do_bench(dw_backward)
torch.cuda.synchronize()
print(f"Triton Backward dw only: {perf_dw(ms_dw):.2f} TFLOPs | Time: {ms_dw:.2f}ms") 

ms_total = triton.testing.do_bench(full_backward)
torch.cuda.synchronize()
print(f"Triton Backward total: {perf_total(ms_total):.2f} TFLOPs | Time: {ms_total:.2f}ms")

print("==========================Benchmark backward end==========================")
