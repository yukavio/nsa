import torch
from nsa.compression_kv import compress_kv

conv1d = torch.nn.functional.conv1d

bs, seqlen, head_dim, kv_num_head = 5, 1024, 128, 2
block_size, block_stride = 64, 16
dtype = torch.bfloat16
device = "cuda"
torch.set_default_device(device)
torch.manual_seed(3)

k = torch.randn(bs * seqlen, kv_num_head, head_dim, dtype=dtype)
v = torch.randn(bs * seqlen, kv_num_head, head_dim, dtype=dtype)
w_k = torch.randn(block_size*head_dim, head_dim, dtype=dtype)
w_v = torch.randn(block_size*head_dim, head_dim, dtype=dtype)
seq_len = torch.Tensor([0] + [seqlen] * bs)
cu_seq_len = torch.cumsum(seq_len, dim=0).to(torch.int32).to(device)

c_k, c_v =  compress_kv(k, v, w_k, w_v, cu_seq_len, block_stride, block_size)



ref_k = torch.zeros_like(c_k)

# torch naive

out_idx = 0
for i in range(bs):
    start_idx = int(cu_seq_len[i])
    end_idx = int(cu_seq_len[i+1])
    seq_len = end_idx - start_idx
    single_k = k[start_idx:end_idx, :, :]
    for w in range((seq_len-block_size)//block_stride):
        w_start = w * block_stride
        w_end = w_start + block_size
        k_window = single_k[w_start:w_end,:, :] # shape: (block_size, head_dim)
        for h in range(kv_num_head):
            single_head_k = k_window[:, h, :] # shape: (seq_len, head_dim)
            single_head_k = single_head_k.reshape(1, -1)
            ref_k[out_idx, h, :] = torch.matmul(single_head_k, w_k)
        out_idx += 1

torch.testing.assert_close(c_k, ref_k, rtol=1e-2, atol=1e-2)



# ref_k = k.reshape(bs, seqlen, kv_num_head, head_dim)
# ref_v = v.reshape(bs, seqlen, kv_num_head, head_dim)

# b_k = ref_k[0].unsqueeze(0).permute(0, 3, 1, 2).reshape(1, head_dim, -1)
# w_k = w_kv[0].reshape(1, 1, kv_num_head*block_size)
# ref_ck = conv1d(b_k, w_k, stride=block_stride, groups=head_dim) 

# print(ref_ck)
# pid (0, 0, 0) idx ( 1,  32): 49.190071
# pid (0, 0, 0) idx ( 1,  33): 121.744659
# pid (0, 0, 0) idx ( 1,  34): -20.112074
# pid (0, 0, 0) idx ( 1,  35): -2.807146
# pid (0, 0, 0) idx ( 1,  36): -66.700356
# pid (0, 0, 0) idx ( 1,  37): -7.357799
# pid (0, 0, 0) idx ( 1,  38): -130.650192
# pid (0, 0, 0) idx ( 1,  39): -48.315445
# pid (0, 0, 0) idx ( 1,  40): -17.515280
# pid (0, 0, 0) idx ( 1,  41): 122.883614
# pid (0, 0, 0) idx ( 1,  42): -92.868103
# pid (0, 0, 0) idx ( 1,  43): -34.701790
# pid (0, 0, 0) idx ( 1,  44): -3.129343
# pid (0, 0, 0) idx ( 1,  45): -57.839584
# pid (0, 0, 0) idx ( 1,  46): 87.838249
