import torch
from nsa.nsa import CompressionAttn
from nsa.torch_attention import attention_ref as attn_func

bs, num_q_head, num_kv_head, head_dim = 2, 32, 2, 128
compress_block_size, compress_block_stride = 64, 16
selection_block, selected_block_count = 64, 32
seq_len = 1024

dtype = torch.bfloat16
device = "cuda"
torch.set_default_device(device)
torch.set_default_dtype(dtype)

q = torch.randn(bs*seq_len, num_q_head, head_dim)
k = torch.randn(bs*seq_len, num_kv_head, head_dim)
v = torch.randn(bs*seq_len, num_kv_head, head_dim)
t = torch.Tensor([0] + [seq_len] * bs)
cu_seq_len = torch.cumsum(t, dim=0).to(torch.int32).to(device)

compressor = CompressionAttn(compress_block_stride, compress_block_size, head_dim, None, device='cuda', dtype=torch.bfloat16)


inp = {'q': q, 'k': k, 'v': v, 'causal': True, 'softmax_scale': 1.0, 'cu_seqlens_k': cu_seq_len}

out, attn_score = compressor(**inp)




# b, num_q_head, t1, t2

kernel_size = selection_block // compress_block_stride + 1
padding = compress_block_size // compress_block_stride - 1
stride = selection_block // compress_block_stride
pooler = torch.nn.AvgPool1d(kernel_size, stride, padding)
score = attn_score.reshape(bs, num_kv_head, -1, *attn_score.shape[-2:])
score = score.sum(2)
import pdb; pdb.set_trace()
a = pooler(score)



