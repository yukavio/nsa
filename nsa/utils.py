
import torch
import triton
import triton.language as tl


#Copy and edited from https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html#sphx-glr-getting-started-tutorials-02-fused-softmax-py
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx.to(tl.int64) * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx.to(tl.int64) * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

@triton.jit
def inplcace_softmax_kernel(output_ptr, input_ptr, TQ, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program

    base_off = tl.program_id(0) * tl.num_programs(1) * TQ * n_cols + tl.program_id(1) * TQ * n_cols
    # base_off = 0
    row_start = tl.program_id(2)
    row_step = tl.num_programs(2)
    for row_idx in tl.range(row_start, TQ, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_offset = row_idx.to(tl.int64) * n_cols + base_off
        row_start_ptr = input_ptr + row_offset
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_offset
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


@triton.jit
def fast_softmax_kernel(output_ptr, input_ptr, M_i, TQ, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr, sm_scale: tl.constexpr, compress_block_size: tl.constexpr,
                   causal: tl.constexpr):
    # starting row of the program

    bid = tl.program_id(0).to(tl.int64)
    hid = tl.program_id(1).to(tl.int64)
    base_off = bid * tl.num_programs(1) * TQ * n_cols + hid * TQ * n_cols
    m_off = bid * tl.num_programs(1) * TQ + hid
    row_start = tl.program_id(2)
    row_step = tl.num_programs(2)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    for row_idx in tl.range(row_start, TQ, row_step, num_stages=num_stages):
        row_offset = row_idx.to(tl.int64) * n_cols + base_off
        output_row_start_ptr = output_ptr + row_offset
        output_ptrs = output_row_start_ptr + col_offsets
        if row_idx < compress_block_size and causal:
            tl.store(output_ptrs, tl.zeros([BLOCK_SIZE], 
                        dtype=output_ptr.type.element_ty), mask=mask)
        else:
            m_i_offset = row_idx.to(tl.int64) * tl.num_programs(1) + m_off
            row_start_ptr = input_ptr + row_offset
            input_ptrs = row_start_ptr + col_offsets
            row = tl.load(input_ptrs, mask=mask, other=0)*sm_scale
            denominator = tl.math.exp2(tl.load(M_i + m_i_offset))
            numerator = tl.exp(row.to(tl.float32))
            softmax_output = numerator / denominator
            tl.store(output_ptrs, softmax_output, mask=mask)