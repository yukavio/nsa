import torch

from nsa import selection_attention
from nsa.nsa import NSAAttention

# from nsa.nsa import CompressionAttn
from nsa.torch_attention import attention_ref as attn_func
from nsa.torch_attention import torch_attntion

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

o = attn(q, k, v, cu_seq_len, 0, causal=True)
assert not torch.isnan(o).any(), 'forward output has nan.'

loss = o.sum()
loss.backward()


assert not torch.isnan(q.grad).any(), 'q.grad output has nan.'
assert not torch.isnan(k.grad).any(), 'k.grad output has nan.'
assert not torch.isnan(v.grad).any(), 'v.grad output has nan.'