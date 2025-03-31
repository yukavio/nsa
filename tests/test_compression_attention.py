import torch
from nsa.torch_attention import attention_ref, torch_attntion
from nsa.triton_attention import flash_attn_func
from nsa.compression_kv import KVCompressor


bs, num_q_head, num_kv_head, head_dim = 1, 64, 4, 128
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


compressor = KVCompressor(
            compress_block_stride, compress_block_size, head_dim, device, dtype
        )

ck, cv, compress_cu_kv_len = compressor(k, v, t, num_q_head//k.shape[1])

q = q.reshape(bs, seq_len, num_q_head, head_dim)

# ref_o, ref_s = attention_ref(q, ck, cv, compress_block_stride, compress_block_size, causal=True, scale=1.0)
# o, s = flash_attn_func(q, ck, cv, compress_block_stride, compress_block_size, None, True, 1.0)


#ref_o, ref_s = attention_ref(q, ck, cv, compress_block_stride, compress_block_size, causal=True, scale=None)
o, s = flash_attn_func(q, ck, cv, compress_block_stride, compress_block_size, True, None)

# print(ref_o.isnan().any())
# print(o.isnan().sum())
# #import pdb; pdb.set_trace()

# torch.testing.assert_close(o, ref_o, rtol=1e-2, atol=1e-2)
# torch.testing.assert_close(s, ref_s, rtol=1e-2, atol=1e-2)

