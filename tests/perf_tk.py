import torch
import numpy as np
import thunderkittens as tk
import matplotlib.pyplot as plt
from collections import defaultdict
import triton
from flash_attn_interface import flash_attn_func as flash_attn_func_v3
from test_compression_attention import safe_all_close

def flops(batch, seqlen, nheads, headdim, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    flop = flop / 1e12
    time = time / 1e3
    return flop / time

def convert_layout(x):
    #return x.permute(0, 2, 1, 3).detach().contiguous().requires_grad_()
    return x.detach().contiguous().requires_grad_()

def bwd(*inputs, y, grad):
    for x in inputs:
        if isinstance(x, torch.Tensor):
            x.grad = None
    y.backward(grad, retain_graph=True)

def benchmark_attention(configurations):
    results = {
        'fwd': defaultdict(list),
        'bwd': defaultdict(list)
    }
    
    for B, H, N, D, causal in configurations:
        print("=" * 60)
        print(f"Timing forward and backward pass for B={B}, H={H}, N={N}, D={D}, causal={causal}")

        q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device='cuda', requires_grad=False).contiguous()
        k = torch.randn(B, N//32, H//16, D, dtype=torch.bfloat16, device='cuda', requires_grad=False).contiguous()
        v = torch.randn(B, N//32, H//16, D, dtype=torch.bfloat16, device='cuda', requires_grad=False).contiguous()

        grad_output = torch.randn_like(q, requires_grad=False).contiguous()
        
        # Flash-Attn
        fla_fwd = triton.testing.do_bench(lambda: flash_attn_func_v3(q, k, v, causal=causal))
        fla_fwd_tflops = efficiency(flops(B, N, H, D, causal, 'fwd'), fla_fwd)

        # TK
        tk_fwd = triton.testing.do_bench(lambda: tk.mha_forward(q, k, v, causal))
        tk_fwd_tflops = efficiency(flops(B, N, H, D, causal, 'fwd'), tk_fwd)
        tk_o, lse = tk.mha_forward(q, k, v, causal)

        tk_bwd = triton.testing.do_bench(lambda: tk.mha_backward(q, k, v, tk_o, lse, grad_output, causal))
        tk_bwd_tflops = efficiency(flops(B, N, H, D, causal, 'bwd'), tk_bwd)
        q_grad, k_grad, v_grad = tk.mha_backward(q, k, v, tk_o, lse, grad_output, causal)

        # test fwd accuracy
        q = convert_layout(q)
        k = convert_layout(k)
        v = convert_layout(v)
        fla_o, lse_ref = flash_attn_func_v3(q, k, v, causal=causal)

        torch.testing.assert_close(fla_o, tk_o, rtol=5e-3, atol=5e-3)
        lse = lse.squeeze(-1) / (-11.3137 if D == 128 else -8)
        torch.testing.assert_close(lse, lse_ref, rtol=5e-3, atol=5e-3)
        
        
        # test bwd accuracy
        bwd(q, k, v, y=fla_o, grad=grad_output)
        fla_bwd = triton.testing.do_bench(lambda: bwd(q, k, v, y=fla_o, grad=grad_output))
        fla_bwd_tflops = efficiency(flops(B, N, H, D, causal, 'bwd'), fla_bwd)
        
        safe_all_close(q.grad, q_grad.to(torch.bfloat16), rtol=5e-3, atol=5e-3)
        safe_all_close(k.grad, k_grad.to(torch.bfloat16), rtol=5e-3, atol=5e-3)
        safe_all_close(v.grad, v_grad.to(torch.bfloat16), rtol=5e-3, atol=5e-3)
    
        print(f"FWD: TK:{tk_fwd:.2f}  FLA3: {fla_fwd:.2f}")
        print(f"TFLOPS: TK:{tk_fwd_tflops}  FLA3: {fla_fwd_tflops:.2f}")
        print("-" * 60)
        print(f"BWD: TK:{tk_bwd:.2f}  FLA3: {fla_bwd:.2f}")
        print(f"TFLOPS: TK:{tk_bwd_tflops}  FLA3: {fla_bwd_tflops:.2f}")
        print("=" * 60)
        
    
    return results


# Example list of configurations to test
configurations = [
    (16, 32, 4096, 128, False),
    (16, 32, 4096,  64, False),
    #(16, 16, 1920*16,  128, True),
    #(16, 32, 1920*16,  64, True),
]

results = benchmark_attention(configurations)
# plot_results(results)
