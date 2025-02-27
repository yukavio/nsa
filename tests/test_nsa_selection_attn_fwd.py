import torch
import triton
from flash_attn import flash_attn_varlen_func

from nsa import attention

bs, seqlen, head_dim, q_num_head, kv_num_head = 32, 4096, 128, 32, 2
block_size = 64
block_num = 16
dtype = torch.bfloat16
device = "cuda"
torch.set_default_device(device)
torch.manual_seed(1)

assert seqlen % block_size == 0
assert block_num * block_size <= seqlen


q = torch.randn(bs * seqlen, q_num_head, head_dim, dtype=dtype)
k = torch.randn(bs * seqlen, kv_num_head, head_dim, dtype=dtype)
v = torch.randn(bs * seqlen, kv_num_head, head_dim, dtype=dtype)
seq_len = torch.Tensor([0] + [seqlen] * bs)
cu_seq_len = torch.cumsum(seq_len, dim=0).to(torch.int32).to(device)

total_block_num = seqlen // block_size
select_id = torch.arange(block_num).repeat(bs, 1)
mask = torch.ones((bs, total_block_num), dtype=torch.bool)
temp = torch.zeros_like(mask)
mask.scatter_(1, select_id, temp)

ref_k = k.clone().reshape(bs, -1, block_size, kv_num_head, head_dim)
ref_v = v.clone().reshape(bs, -1, block_size, kv_num_head, head_dim)
ref_k[mask] = 0
ref_v[mask] = 0

ref_k = ref_k.reshape(-1, kv_num_head, head_dim).contiguous()
ref_v = ref_v.reshape(-1, kv_num_head, head_dim).contiguous()
ref_q = q

select_id = select_id.flatten().contiguous()

for causal in [True, False]:
    ref_out = flash_attn_varlen_func(
        q,
        ref_k,
        ref_v,
        cu_seq_len,
        cu_seq_len,
        seqlen,
        seqlen,
        causal=causal,
        softmax_scale=1.0,
    )
    out = attention(q, k, v, cu_seq_len, select_id, causal, 1.0, block_size, block_num)
    torch.testing.assert_close(ref_out, out, atol=1e-2, rtol=2e-2)

perf = (
    lambda ms: 2.0 * bs * q_num_head * seqlen * seqlen * head_dim * 1e-12 / (ms * 1e-3)
)
causal = True

ms = triton.testing.do_bench(
    lambda: flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seq_len,
        cu_seq_len,
        seqlen,
        seqlen,
        causal=causal,
        softmax_scale=1.0,
    )
)


print("flash-attn2 {} TFlops".format(perf(ms)))

ms = triton.testing.do_bench(
    lambda: attention(
        q, k, v, cu_seq_len, select_id, causal, 1.0, block_size, block_num
    )
)

print("nsa {} TFlops".format(perf(ms)))
