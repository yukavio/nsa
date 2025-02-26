import torch
import triton
from flash_attn import flash_attn_varlen_func

from nsa import attention

bs, seqlen, head_dim, q_num_head, kv_num_head = 32, 1024, 128, 64, 4
dtype = torch.bfloat16
device = "cuda"
torch.set_default_device(device)
torch.manual_seed(2)


q = torch.randn(bs * seqlen, q_num_head, head_dim, dtype=dtype)
k = torch.randn(bs * seqlen, kv_num_head, head_dim, dtype=dtype)
v = torch.randn(bs * seqlen, kv_num_head, head_dim, dtype=dtype)
seq_len = torch.Tensor([0] + [seqlen] * bs)
cu_seq_len = torch.cumsum(seq_len, dim=0).to(torch.int32).to(device)

for causal in [True, False]:
    ref_out = flash_attn_varlen_func(
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

    out = attention(q, k, v, cu_seq_len, causal, 1.0)
    # torch.testing.assert_close(ref_out, out, atol=1e-2, rtol=2e-2)

perf = (
    lambda ms: 2.0 * bs * q_num_head * seqlen * seqlen * head_dim * 1e-12 / (ms * 1e-3)
)

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

ms = triton.testing.do_bench(lambda: attention(q, k, v, cu_seq_len, causal, 1.0))

print("nsa {} TFlops".format(perf(ms)))
