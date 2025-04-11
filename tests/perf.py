import torch

from nsa.nsa import NSAAttention
import triton
import time
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from torch.profiler import profile, record_function, ProfilerActivity
torch.manual_seed(10)

bs, num_q_head, num_kv_head, head_dim = 1, 64, 4, 128
compress_block_size, compress_block_stride = 64, 16
selection_block, selected_block_count = 64, 32
seq_len = 1024*32

dtype = torch.bfloat16
device = "cuda"
torch.set_default_device(device)
torch.set_default_dtype(dtype)

q = torch.randn(bs*seq_len, num_q_head, head_dim, requires_grad=True)
k = torch.randn(bs*seq_len, num_kv_head, head_dim, requires_grad=True)
v = torch.randn(bs*seq_len, num_kv_head, head_dim, requires_grad=True)
t = torch.Tensor([0] + [seq_len] * bs)
cu_seq_len = torch.cumsum(t, dim=0).to(torch.int32).to(device)


attn = NSAAttention(head_dim, 0, True, None, 0, device=device, dtype=dtype)

nsa_fwd = lambda : attn(q, k, v, cu_seq_len, 0, causal=True)
flash_fwd = lambda : flash_attn_varlen_func(q, k, v, cu_seq_len, cu_seq_len, seq_len, seq_len)


def test(fwd_func, name):
    for i in range(5):
        o = fwd_func()
        loss = (o*o).sum()
        loss.backward()

    torch.cuda.synchronize()

    repeat=10
    forward_time = 0
    backward_time = 0
    for i in range(repeat):
        torch.cuda.synchronize()
        start = time.time()
        o = fwd_func()
        torch.cuda.synchronize()
        forward_time += time.time()-start
        loss = (o*o).sum()
        torch.cuda.synchronize()
        start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        backward_time += time.time()-start
    
    forward_time /= repeat / 1e3
    backward_time /= repeat / 1e3

    total_time = forward_time + backward_time

    # d_model = num_q_head * head_dim
    # total_flops = 2 * 2 * (seq_len ** 2) * d_model * num_kv_head
    # tflops = total_flops / (total_time * 1e-3) / 1e12

    ms_per_iter = total_time
    print('********* ',name, ' ***********')
    print(f"Forward: {forward_time:.3f}ms | Backward: {backward_time:.3f}ms | Total: {total_time:.3f}ms")
    sort_by_keyword = device + "_time_total"
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        o = fwd_func()
    print('Forward profile')
    print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))

    loss = (o*o).sum()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        loss.backward()
    print('Backward profile')
    print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))

    #print(f"Estimated TFLOPs/s: {tflops:.2f}")

test(nsa_fwd, 'NSA')
test(flash_fwd, 'FLASH ATTN')

# o = nsa_fwd()
# loss = (o*o).sum()
# sort_by_keyword = device + "_time_total"
# with profile(activities=[ProfilerActivity.CUDA]) as prof:
#     loss.backward()
# print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))

# prof.export_chrome_trace("trace.json")



# test(nsa_fwd)
# test(flash_fwd)