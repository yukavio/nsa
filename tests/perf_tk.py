import torch
import numpy as np
import thunderkittens as tk
import matplotlib.pyplot as plt
from collections import defaultdict
import triton
import os
import time
from flash_attn_interface import flash_attn_func as flash_attn_func_v3

def flops(batch, seqlen, nheads, headdim, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    flop = flop / 1e12
    time = time / 1e3
    return flop / time

def convert_layout(x):
    return x.permute(0, 2, 1, 3).detach().contiguous().requires_grad_()

def benchmark_attention(configurations):
    results = {
        'fwd': defaultdict(list),
        'bwd': defaultdict(list)
    }
    
    for B, H, N, D, causal in configurations:
        print("=" * 60)
        print(f"Timing forward and backward pass for B={B}, H={H}, N={N}, D={D}, causal={causal}")

        q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device='cuda', requires_grad=False).contiguous()
        k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device='cuda', requires_grad=False).contiguous()
        v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device='cuda', requires_grad=False).contiguous()
        t = torch.Tensor([0] + [N] * B).to('cuda')
        cu_seq_len = torch.cumsum(t, dim=0).to(torch.int32).to('cuda')

        o           = torch.zeros_like(q).contiguous()
        grad_output = torch.randn_like(q, requires_grad=False).contiguous()

        # TK
        o, l_vec = tk.mha_forward(q, k, v, causal)
        tk_fwd = triton.testing.do_bench(lambda: tk.mha_forward(q, k, v, causal))
        tk_fwd_tflops = efficiency(flops(B, N, H, D, causal, 'fwd'), tk_fwd)

        tk_bwd = triton.testing.do_bench(lambda: tk.mha_backward(q, k, v, o, l_vec, grad_output, causal))
        tk_bwd_tflops = efficiency(flops(B, N, H, D, causal, 'bwd'), tk_bwd)
        
        
        # Flash-Attn
        q = convert_layout(q)
        k = convert_layout(k)
        v = convert_layout(v)
        fla_fwd = triton.testing.do_bench(lambda: flash_attn_func_v3(q, k, v, causal=causal))
        fla_fwd_tflops = efficiency(flops(B, N, H, D, causal, 'fwd'), fla_fwd)
        
        res = flash_attn_func_v3(q, k, v, causal=causal)
        o = res[0]
        #import pdb; pdb.set_trace()
        def bwd(*inputs, y, grad):
            for x in inputs:
                if isinstance(x, torch.Tensor):
                    x.grad = None
            y.backward(grad, retain_graph=True)
        grad = torch.randn_like(o)
        fla_bwd = triton.testing.do_bench(lambda: bwd(q, k, v, y=o, grad=grad))
        fla_bwd_tflops = efficiency(flops(B, N, H, D, causal, 'bwd'), fla_bwd)
    
    
        print(f"FWD: TK:{tk_fwd:.2f}  FLA3: {fla_fwd:.2f}")
        print(f"TFLOPS: TK:{tk_fwd_tflops}  FLA3: {fla_fwd_tflops:.2f}")
        print("-" * 60)
        print(f"BWD: TK:{tk_bwd:.2f}  FLA3: {fla_bwd:.2f}")
        print(f"TFLOPS: TK:{tk_bwd_tflops}  FLA3: {fla_bwd_tflops:.2f}")
        print("=" * 60)
        
    
    return results


# Example list of configurations to test
configurations = [
    (16, 16, 1024, 128, True),
    (16, 16, 1024*32,  128, True),
    (16, 16, 1024*32,  64, True),
]

results = benchmark_attention(configurations)
# plot_results(results)
