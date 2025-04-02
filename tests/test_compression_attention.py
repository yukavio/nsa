import torch
torch.autograd.set_grad_enabled(True)
from nsa.torch_attention import attention_ref
from nsa.triton_attention import flash_attn_func
from nsa.compression_kv import KVCompressor


bs, num_q_head, num_kv_head, head_dim = 1, 4, 4, 128
compress_block_size, compress_block_stride = 64, 16
selection_block, selected_block_count = 64, 32
seq_len = 1024*16



dtype = torch.bfloat16
device = "cuda"
torch.set_default_device(device)
torch.set_default_dtype(dtype)
torch.manual_seed(9)

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

q_t = q.reshape(bs, seq_len, num_q_head, head_dim)
q_ref_t = q_ref.reshape(bs, seq_len, num_q_head, head_dim)

# compressor = KVCompressor(
#             compress_block_stride, compress_block_size, head_dim, device, dtype
#         )

# ck, cv, compress_cu_kv_len = compressor(k, v, t, num_q_head//k.shape[1])
# ck_ref, cv_ref, compress_cu_kv_len = compressor(k_ref, v_ref, t, num_q_head//k.shape[1])

# ref_o, ref_s = attention_ref(q_ref, ck_ref, cv_ref, compress_block_stride, compress_block_size, causal=True, scale=None)
# o, s = flash_attn_func(q, ck, cv, compress_block_stride, compress_block_size, True, None)

# assert not o.isnan().any()
# torch.testing.assert_close(o, ref_o, rtol=1e-2, atol=1e-2)


k_t = k.reshape(bs, seq_len, num_kv_head, head_dim)
v_t = v.reshape(bs, seq_len, num_kv_head, head_dim)
k_ref_t = k_ref.reshape(bs, seq_len, num_kv_head, head_dim)
v_ref_t = v_ref.reshape(bs, seq_len, num_kv_head, head_dim)


ref_o, ref_s = attention_ref(q_ref_t, k_ref_t, v_ref_t, compress_block_stride, compress_block_size, causal=False, scale=None)

ref_loss = (ref_o*ref_o).sum()
ref_loss.backward()
# print(ref_loss)
# print(k_ref.grad)



o, s = flash_attn_func(q_t, k_t, v_t, compress_block_stride, compress_block_size, False, None)
torch.testing.assert_close(o, ref_o, rtol=1e-2, atol=1e-2)
loss = (o*o).sum()
loss.backward()


# print(q.grad)
# print(q_ref.grad)


torch.testing.assert_close(q.grad, q_ref.grad, rtol=1e-2, atol=1e-2)
torch.testing.assert_close(k.grad, k_ref.grad, rtol=1e-2, atol=1e-2)
torch.testing.assert_close(v.grad, v_ref.grad, rtol=1e-2, atol=1e-2)

print("Test passed!")



def test_no_causal():
    k_t = k.reshape(bs, seq_len, num_kv_head, head_dim)
    v_t = v.reshape(bs, seq_len, num_kv_head, head_dim)
    k_ref_t = k_ref.reshape(bs, seq_len, num_kv_head, head_dim)
    v_ref_t = v_ref.reshape(bs, seq_len, num_kv_head, head_dim)


    ref_o, ref_s = attention_ref(q_ref_t, k_ref_t, v_ref_t, compress_block_stride, compress_block_size, causal=False, scale=None)
    ref_loss = (ref_o*ref_o).sum()
    ref_loss.backward()

    o, s = flash_attn_func(q_t, k_t, v_t, compress_block_stride, compress_block_size, False, None)
    torch.testing.assert_close(o, ref_o, rtol=1e-2, atol=1e-2)
    loss = (o*o).sum()
    loss.backward()

    torch.testing.assert_close(v.grad, v_ref.grad, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(k.grad, k_ref.grad, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(q.grad, q_ref.grad, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(s, ref_s, rtol=1e-2, atol=1e-2)

test_no_causal()
print("no causal test passed!")