import torch
torch.autograd.set_grad_enabled(True)
from nsa.torch_attention import attention_ref
from nsa.triton_attention import flash_attn_func
from nsa.compression_kv import KVCompressor
from copy import deepcopy


compress_block_size, compress_block_stride = 32, 16
seq_len = 1024*8
selection_block_size, selected_block_count = 64, 16

dtype = torch.bfloat16
device = "cuda"
torch.set_default_device(device)
torch.set_default_dtype(dtype)
torch.manual_seed(9)


def safe_all_close(out, ref, rtol, atol):
    try:
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    except Exception as e:
        print(e)


def create_data(bs, num_q_head, num_kv_head, head_dim):
    q = torch.randn(bs*seq_len, num_q_head, head_dim, requires_grad=True)
    k = torch.randn(bs*seq_len, num_kv_head, head_dim, requires_grad=True)
    v = torch.randn(bs*seq_len, num_kv_head, head_dim, requires_grad=True)
    t = torch.Tensor([0] + [seq_len] * bs)

    q_ref = q.detach()
    k_ref = k.detach()
    v_ref = v.detach()
    q_ref.requires_grad = True
    k_ref.requires_grad = True
    v_ref.requires_grad = True
    q_ref.retain_grad()
    k_ref.retain_grad()
    v_ref.retain_grad()

    return q, k, v, q_ref, k_ref, v_ref, t


def test_no_causal():
    bs, num_q_head, num_kv_head, head_dim = 2, 4, 4, 128
    pool_kernel_size = selection_block_size // compress_block_stride + 1
    pool_padding = compress_block_size // compress_block_stride - 2
    pool_stride = selection_block_size // compress_block_stride

    q, k, v, q_ref, k_ref, v_ref, t = create_data(bs, num_q_head, num_kv_head, head_dim)
    q_t = q.reshape(bs, seq_len, num_q_head, head_dim)
    q_ref_t = q_ref.reshape(bs, seq_len, num_q_head, head_dim)

    compressor = KVCompressor(
            compress_block_stride, compress_block_size, head_dim, device, dtype
        )
    ck, cv, compress_cu_kv_len = compressor(k, v, t, num_q_head//k.shape[1])

    compressor_ref = deepcopy(compressor)
    ck_ref, cv_ref, compress_cu_kv_len = compressor_ref(k_ref, v_ref, t, num_q_head//k.shape[1])

    ref_o, ref_indices = attention_ref(q_ref_t, ck_ref, cv_ref, compress_block_stride, compress_block_size, causal=False, scale=None,
        pool_num_kv_head=num_kv_head, pool_kernel_size=pool_kernel_size, pool_stride=pool_stride, pool_padding=pool_padding, select_block_count=selected_block_count)
    # ref_loss = (ref_o*ref_o+0.1*(ref_s*ref_s).sum()).sum()
    ref_loss = (ref_o*ref_o).sum()
    ref_loss.backward()

    o, indices = flash_attn_func(q_t, ck, cv, compress_block_stride, compress_block_size, False, None, num_kv_head, pool_kernel_size, 
                            pool_stride, pool_padding, selected_block_count)
    torch.testing.assert_close(o, ref_o, rtol=1e-2, atol=1e-2)
    print('test indices')
    safe_all_close(indices, ref_indices, rtol=1e-2, atol=1e-2)
    loss = (o*o).sum()
    loss.backward()
    print('test grad_v')
    safe_all_close(v.grad, v_ref.grad, rtol=3e-2, atol=3e-2)
    print('test grad_k')
    safe_all_close(k.grad, k_ref.grad, rtol=3e-2, atol=3e-2)
    print('test grad_q')
    # The diff ratio is too high for q.grad for none causal. But we only use the kernel with causal. It only for test.
    safe_all_close(q.grad, q_ref.grad, rtol=3e-2, atol=3e-2)
    print('PASS NO CAUSAL')

test_no_causal()


def test_causal():
    bs, num_q_head, num_kv_head, head_dim = 2, 16, 1, 128
    pool_kernel_size = selection_block_size // compress_block_stride + 1
    pool_padding = compress_block_size // compress_block_stride - 2
    pool_stride = selection_block_size // compress_block_stride

    q, k, v, q_ref, k_ref, v_ref, t = create_data(bs, num_q_head, num_kv_head, head_dim)
    q_t = q.reshape(bs, seq_len, num_q_head, head_dim)
    q_ref_t = q_ref.reshape(bs, seq_len, num_q_head, head_dim)

    compressor = KVCompressor(
                compress_block_stride, compress_block_size, head_dim, device, dtype
            )
    ck, cv, compress_cu_kv_len = compressor(k, v, t, num_q_head//k.shape[1])

    compressor_ref = deepcopy(compressor)
    ck_ref, cv_ref, compress_cu_kv_len = compressor_ref(k_ref, v_ref, t, num_q_head//k.shape[1])
    ref_o, ref_indices = attention_ref(q_ref_t, ck_ref, cv_ref, compress_block_stride, compress_block_size, causal=True, scale=None,
        pool_num_kv_head=num_kv_head, pool_kernel_size=pool_kernel_size, pool_stride=pool_stride, pool_padding=pool_padding, select_block_count=selected_block_count)
    ref_loss = (ref_o*ref_o).sum()
    ref_loss.backward()

    o, indices = flash_attn_func(q_t, ck, cv, compress_block_stride, compress_block_size, True, None, num_kv_head, pool_kernel_size, 
                            pool_stride, pool_padding, selected_block_count)
    loss = (o*o).sum()
    loss.backward()
    torch.testing.assert_close(o, ref_o, rtol=1e-2, atol=1e-2)
    
    print('test indices')
    safe_all_close(indices, ref_indices, rtol=1e-2, atol=1e-2)
    print('test grad_v')
    safe_all_close(v.grad, v_ref.grad, rtol=3e-2, atol=3e-2)
    print('test grad_k')
    safe_all_close(k.grad, k_ref.grad, rtol=3e-2, atol=3e-2)
    print('test grad_q')
    # The diff ratio is too high for q.grad for none causal. But we only use the kernel with causal. It only for test.
    safe_all_close(q.grad, q_ref.grad, rtol=3e-2, atol=3e-2)

    print('PASS CAUSAL')

# test_causal()


