import torch
from nsa.compression_kv import compress_kv, calc_compressed_len
import triton

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

def compute_reference_kv(input_tensor: torch.Tensor, 
                        weight: torch.Tensor,
                        cu_seq_len: torch.Tensor,
                        block_size: int,
                        block_stride: int) -> torch.Tensor:
    num_heads = input_tensor.size(1)
    result_list = []
    batch_size = len(cu_seq_len) - 1
    
    for i in range(batch_size):
        start_idx = int(cu_seq_len[i])
        end_idx = int(cu_seq_len[i+1])
        seq_len = end_idx - start_idx
        single_seq = input_tensor[start_idx:end_idx, :, :]
        
        num_windows = calc_compressed_len(seq_len, block_stride, block_size)
        for w in range(num_windows):
            w_start = w * block_stride
            w_end = w_start + block_size
            window = single_seq[w_start:w_end, :, :]
            
            head_results = []
            for h in range(num_heads):
                single_head = window[:, h, :]
                single_head_flat = single_head.reshape(1, -1)
                head_result = torch.matmul(single_head_flat, weight)
                head_results.append(head_result)
            
            window_result = torch.stack(head_results, dim=1)
            result_list.append(window_result)
    
    return torch.cat(result_list, dim=0)

target = torch.randn_like(c_k)

# test tflops 
def calculate_flops(BATCH_SIZE, SEQ_LENGTH, HEAD_DIM, KV_NUM_HEADS, BLOCK_SIZE, BLOCK_STRIDE):
    compressed_len = max(0, (SEQ_LENGTH - BLOCK_SIZE + BLOCK_STRIDE) // BLOCK_STRIDE)
    total_out_tokens = BATCH_SIZE * compressed_len
    
    single_fwd_flops = total_out_tokens * KV_NUM_HEADS * (2 * BLOCK_SIZE * HEAD_DIM * HEAD_DIM)
    fwd_flops = 2 * single_fwd_flops
    
    single_dw_flops = total_out_tokens * KV_NUM_HEADS * (2 * BLOCK_SIZE * HEAD_DIM * HEAD_DIM)
    single_dx_flops = total_out_tokens * KV_NUM_HEADS * (2 * BLOCK_SIZE * HEAD_DIM * HEAD_DIM)
    
    dw_flops = 2 * single_dw_flops 
    dx_flops = 2 * single_dx_flops 
    
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
for _ in range(10):
    compress_kv(k, v, w_k, w_v, cu_seq_len, block_stride, block_size)
    compute_reference_kv(k, w_k, cu_seq_len, block_size, block_stride)
    
perf = lambda ms: fwd_flops * 1e-12 / (ms * 1e-3)

ms_forward_triton = triton.testing.do_bench(
    lambda: compress_kv(k, v, w_k, w_v, cu_seq_len, block_stride, block_size)
)

print(f"Triton Forward: {perf(ms_forward_triton):.2f} TFLOPs | Time: {ms_forward_triton:.2f}ms")

ms_forward_torch = triton.testing.do_bench(
    lambda: compute_reference_kv(k, w_k, cu_seq_len, block_size, block_stride)
)

print(f"Torch Forward: {perf(ms_forward_torch):.2f} TFLOPs | Time: {ms_forward_torch:.2f}ms")

print("==========================Benchmark forward end==========================")

print("==========================Benchmark backward start==========================")
# warm up
for _ in range(10):
    c_k, c_v = compress_kv(k, v, w_k, w_v, cu_seq_len, block_stride, block_size)
    c_loss = torch.mean((c_k - target) ** 2)
    c_loss.backward(retain_graph=True)
    w_k.grad = None
    k.grad = None
    w_v.grad = None
    v.grad = None
    
    ref_k = compute_reference_kv(k, w_k, cu_seq_len, block_size, block_stride)
    ref_loss = torch.mean((ref_k - target) ** 2)
    ref_loss.backward(retain_graph=True)
    w_k.grad = None
    k.grad = None
    w_v.grad = None
    v.grad = None

total_bwd_flops = 2 * (dw_flops + dx_flops)

def full_backward():
    loss = torch.mean((c_k - target) ** 2)
    loss.backward(retain_graph=True)
    w_k.grad = None
    k.grad = None
    w_v.grad = None
    v.grad = None
    

def dw_backward():
    loss = torch.mean((c_k - target) ** 2)
    loss.backward(inputs=[w_k, w_v], retain_graph=True)
    w_k.grad = None
    k.grad = None
    w_v.grad = None
    v.grad = None

def dx_backward():
    loss = torch.mean((c_k - target) ** 2)
    loss.backward(inputs=[k, v], retain_graph=True)
    w_k.grad = None
    k.grad = None
    w_v.grad = None
    v.grad = None

def torch_backward():
    loss = torch.mean((ref_k - target) ** 2)
    loss.backward(inputs=[w_k, w_v], retain_graph=True)
    w_k.grad = None
    k.grad = None
    w_v.grad = None
    v.grad = None

def torch_dw_backward():
    loss = torch.mean((ref_k - target) ** 2)
    loss.backward(inputs=[w_k, w_v], retain_graph=True)
    w_k.grad = None
    k.grad = None
    w_v.grad = None
    v.grad = None
    
def torch_dx_backward():
    loss = torch.mean((ref_k - target) ** 2)
    loss.backward(inputs=[k, v], retain_graph=True)
    w_k.grad = None
    k.grad = None
    w_v.grad = None
    v.grad = None


perf_dw = lambda ms: dw_flops * 1e-12 / (ms * 1e-3)
perf_dx = lambda ms: dx_flops * 1e-12 / (ms * 1e-3)
perf_total = lambda ms: (dw_flops + dx_flops) * 1e-12 / (ms * 1e-3)


ms_dw = triton.testing.do_bench(dw_backward)
print(f"Triton Backward dw only: {perf_dw(ms_dw):.2f} TFLOPs | Time: {ms_dw:.2f}ms") 

ms_dx = triton.testing.do_bench(dx_backward)
print(f"Triton Backward dx only: {perf_dx(ms_dx):.2f} TFLOPs | Time: {ms_dx:.2f}ms")

ms_total = triton.testing.do_bench(full_backward)
print(f"Triton Backward total: {perf_total(ms_total):.2f} TFLOPs | Time: {ms_total:.2f}ms")

ms_torch_dw = triton.testing.do_bench(torch_dw_backward)
print(f"Torch Backward dw only: {perf_dw(ms_torch_dw):.2f} TFLOPs | Time: {ms_torch_dw:.2f}ms") 

ms_torch_dx = triton.testing.do_bench(torch_dx_backward)
print(f"Torch Backward dx only: {perf_dx(ms_torch_dx):.2f} TFLOPs | Time: {ms_torch_dx:.2f}ms")

ms_torch = triton.testing.do_bench(torch_backward)
print(f"Torch Backward total: {perf_total(ms_torch):.2f} TFLOPs | Time: {ms_torch:.2f}ms")

print("==========================Benchmark backward end==========================")
