import torch

from nsa import selection_attention
from nsa.nsa import NSAAttention as _NSAAttention
from nsa.new_nsa import NSAAttention, MergedNSAAttention, NSAFunctor
from test_compression_attention import safe_all_close
torch.manual_seed(9)

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

ref_attn = _NSAAttention(head_dim, True, None, 0, device=device, dtype=dtype)
cmp_wk = ref_attn.compressor.compressor_k.weight
cmp_wv = ref_attn.compressor.compressor_v.weight

attn_functor = NSAFunctor(head_dim, True, None)
attn_func = MergedNSAAttention

ref_o = ref_attn(q, k, v, cu_seq_len, 0, causal=True)
ref_loss = (ref_o*ref_o).sum()
ref_loss.backward()

assert not torch.isnan(ref_o).any(), 'forward output has nan.'
assert not torch.isnan(q.grad).any(), 'q.grad output has nan.'
assert not torch.isnan(k.grad).any(), 'k.grad output has nan.'
assert not torch.isnan(v.grad).any(), 'v.grad output has nan.'


q_grad_ref = q.grad.detach()
k_grad_ref = k.grad.detach()
v_grad_ref = v.grad.detach()
cmp_wk_ref = cmp_wk.grad.detach()
cmp_wv_ref = cmp_wv.grad.detach()
gating_ref = ref_attn.gating.weight.grad.detach()

del q.grad, k.grad, v.grad, cmp_wv.grad, cmp_wk.grad


# o = attn_func.forward(q, k, v, ref_attn.compressor.compressor_k.weight, ref_attn.compressor.compressor_v.weight, 
#                       ref_attn.gating.weight, cu_seq_len)
o = attn_func.apply(attn_functor, q, k, v, ref_attn.gating.weight, cmp_wk, cmp_wv, 
                    cu_seq_len)
loss = (o*o).sum()
loss.backward()
print('test output')
torch.testing.assert_close(o, ref_o, rtol=1e-2, atol=1e-2)
print('test q_grad')
safe_all_close(q.grad, q_grad_ref)
print('test k_grad')
safe_all_close(k.grad, k_grad_ref)
print('test v_grad')
safe_all_close(v.grad, v_grad_ref)
print('test cmp_wk_grad')
safe_all_close(cmp_wk.grad, cmp_wk_ref)
print('test cmp_wv_grad')
safe_all_close(cmp_wv.grad, cmp_wv_ref)
print('test gating grad')
safe_all_close(ref_attn.gating.weight.grad, gating_ref)
