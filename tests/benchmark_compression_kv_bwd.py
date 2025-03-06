import torch
from nsa.compression_kv import compress_kv, calc_compressed_len


bs, seqlen, head_dim, kv_num_head = 5, 1024, 128, 2
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


def compute_reference_kv(k, w_k, cu_seq_len, block_size, block_stride):
    """Torch实现的参考计算逻辑，支持自动求导"""
    kv_num_head = k.size(1)  # 获取头数
    k_list = []  # 用于收集每个窗口的结果
    bs = len(cu_seq_len) - 1
    
    for i in range(bs):
        start_idx = int(cu_seq_len[i])
        end_idx = int(cu_seq_len[i+1])
        seq_len = end_idx - start_idx
        single_k = k[start_idx:end_idx, :, :]  # [seq_len, H, D]
        
        num_windows = calc_compressed_len(seq_len, block_stride, block_size)
        for w in range(num_windows):
            w_start = w * block_stride
            w_end = w_start + block_size
            # 处理可能的越界，假设允许不足block_size的窗口（不填充）
            k_window = single_k[w_start:w_end, :, :]  # [有效长度, H, D]
            
            head_results = []
            for h in range(kv_num_head):
                single_head_k = k_window[:, h, :]  # [有效长度, D]
                single_head_k_flat = single_head_k.reshape(1, -1)  # [1, 有效长度*D]
                head_k = torch.matmul(single_head_k_flat, w_k)  # [1, d]
                head_results.append(head_k)
            
            window_k = torch.stack(head_results, dim=1)
            k_list.append(window_k)
    
    ref_k = torch.cat(k_list, dim=0)  # [total_windows, H, d]
    return ref_k

ref_k = compute_reference_kv(k, w_k, cu_seq_len, block_size, block_stride)
torch.testing.assert_close(c_k, ref_k, rtol=1e-2, atol=1e-2)
print("Forward Passed")


# test backward =========

target = torch.randn_like(c_k)
# 计算参考实现的梯度
ref_loss = torch.mean((ref_k - target) ** 2)
ref_loss.backward()
ref_wk_grad = w_k.grad.clone()

w_k.grad = None
k.grad = None

# 计算自定义实现的梯度
c_loss = torch.mean((c_k - target) ** 2)
# c_loss = c_k.sum()
c_loss.backward()
c_wk_grad = w_k.grad.clone()



print("dw_k:", c_wk_grad[0,:10])
print("ref_wk_grad:", ref_wk_grad[0,:10])

torch.testing.assert_close(c_wk_grad, ref_wk_grad, rtol=2e-2, atol=2e-2)
print("Backward Passed")



def calculate_tflops(bs, seqlen, block_size, block_stride, kv_num_head, head_dim):
    num_windows_per_seq = ((seqlen - block_size) // block_stride)
    total_matmuls = bs * num_windows_per_seq * kv_num_head
    flops_per_matmul = 2 * block_size * (head_dim ** 2)
    total_flops = total_matmuls * flops_per_matmul
    return total_flops
# 接下来测量tflops
import triton
total_flops = calculate_tflops(bs, seqlen, block_size, block_stride, kv_num_head, head_dim)

# 预热
print("==========================Benchmark forward start==========================")
for _ in range(10):
    compress_kv(k, v, w_k, w_v, cu_seq_len, block_stride, block_size)
    compute_reference_kv(k, w_k, cu_seq_len, block_size, block_stride)
    
perf = (
    lambda ms: total_flops * 1e-12 / (ms * 1e-3)
)
ms = triton.testing.do_bench(
    lambda: compress_kv(k, v, w_k, w_v, cu_seq_len, block_stride, block_size)
)


print("compress_kv forward {} TFlops".format(perf(ms)))

ms = triton.testing.do_bench(
    lambda: compute_reference_kv(k, w_k, cu_seq_len, block_size, block_stride)
)

print("compute_reference_kv forward {} TFlops".format(perf(ms)))

print("==========================Benchmark forward end==========================")



# print("==========================Benchmark backward start==========================")
# print("==========================Benchmark backward end==========================")