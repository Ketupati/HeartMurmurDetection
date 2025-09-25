from common import *


def pack_ternary_W_to_uint32(W_ternary: torch.Tensor):
    """
    Pack ternary W_ternary [D, D] into uint32 words with 2 bits per entry (16 weights per word).
    Encoding: -1 -> 2 (10b), 0 -> 0 (00b), +1 -> 1 (01b).
    Returns int32 tensor of shape [num_blocks, D], with num_blocks = ceil(D/16).
    """
    assert W_ternary.dim() == 2 and W_ternary.shape[0] == W_ternary.shape[1], "W must be square"
    D = W_ternary.shape[0]
    BLOCK_I = 16
    num_blocks = (D + BLOCK_I - 1) // BLOCK_I

    
    Wbits = torch.zeros_like(W_ternary, dtype=torch.int32)
    Wbits[W_ternary == 1] = 1
    Wbits[W_ternary == -1] = 2

    
    if D % BLOCK_I != 0:
        pad_rows = BLOCK_I - (D % BLOCK_I)
        Wbits = torch.cat([Wbits, torch.zeros((pad_rows, D), dtype=torch.int32)], dim=0)

    
    Wbits = Wbits.view(num_blocks, BLOCK_I, D)

    
    shifts = (2 * torch.arange(BLOCK_I, dtype=torch.int32)).view(BLOCK_I, 1)

    
    W_shifted = Wbits << shifts   # [num_blocks, 16, D]
    W_packed = W_shifted.sum(dim=1)  # [num_blocks, D], dtype int32

    return W_packed





@triton.jit
def bitlinear_kernel(X_ptr, Wpacked_ptr, C_ptr, Y_ptr,
                     B, N, D,
                     stride_xb, stride_xn, stride_xd,
                     stride_wib, stride_wj,
                     stride_yb, stride_yn, stride_yd,
                     BLOCK_D: tl.constexpr):

    b = tl.program_id(0)   # batch
    n = tl.program_id(1)   # seq index
    j_block = tl.program_id(2)  # output feature block index

    BLOCK_I = 16  # fixed packing (2 bits per weight -> 16 weights per 32-bit word)
    offs_j = j_block * BLOCK_D + tl.arange(0, BLOCK_D)         
    mask_j = offs_j < D

    
    X_row = X_ptr + b * stride_xb + n * stride_xn
    Y_row = Y_ptr + b * stride_yb + n * stride_yn

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    num_i_blocks = (D + BLOCK_I - 1) // BLOCK_I
    ib = 0
    while ib < num_i_blocks:
        
        w_word_ptr = Wpacked_ptr + ib * stride_wib
        words = tl.load(w_word_ptr + offs_j * stride_wj, mask=mask_j, other=0)  # int32 vector [BLOCK_D]

        
        k_vec = tl.arange(0, 16)                   # shape [16] - compile-time constant
        i_idx = ib * BLOCK_I + k_vec               # [16]
        mask_i = i_idx < D                          # boolean mask [16]

        
        shifts = 2 * k_vec                         # [16]
        extracted = (words[None, :] >> shifts[:, None]) & 0x3

        # map to sign: 1 -> +1, 2 -> -1, else 0
        is_pos = extracted == 1
        is_neg = extracted == 2
        sign = tl.where(is_pos, 1.0, 0.0) - tl.where(is_neg, 1.0, 0.0)  # [16, BLOCK_D]

        
        x_ptrs = X_row + i_idx * stride_xd            # pointer vector [16] (valid masked by mask_i)
        x_vals = tl.load(x_ptrs, mask=mask_i, other=0.0)  # [16]
        x_vals = x_vals[:, None]                      # [16, 1]

        
        contrib = x_vals * sign                        # [16, BLOCK_D]
        acc += tl.sum(contrib, axis=0)                # sum over 16 -> [BLOCK_D]

        ib += 1

    # scaling vector once per output dim
    c_vals = tl.load(C_ptr + offs_j, mask=mask_j, other=1.0)
    acc *= c_vals

    tl.store(Y_row + offs_j * stride_yd, acc, mask=mask_j)


def bitlinear_forward(X, W_packed, c, block_d=128):
    """
    X: [B, N, D] or [B, D] float32 (CUDA)
    W_packed: [num_i_blocks, D] int32
    c: [D] float32
    """
    was_2d = False
    if X.dim() == 2:
        was_2d = True
        X = X.unsqueeze(1)
    B, N, D = X.shape

    num_i_blocks = (D + 15) // 16
    if W_packed.shape[0] != num_i_blocks or W_packed.shape[1] != D:
        raise ValueError(f"bitlinear_forward: W_packed shape {W_packed.shape} incompatible with D={D}")
    if c.numel() != D:
        raise ValueError(f"bitlinear_forward: c length {c.numel()} does not match D={D}")
    if X.device != W_packed.device or X.device != c.device:
        raise ValueError("bitlinear_forward: X, W_packed and c must be on the same device")

    Y = torch.empty_like(X, dtype=torch.float32)
    grid = (B, N, (D + block_d - 1) // block_d)
    bitlinear_kernel[grid](
        X, W_packed, c, Y,
        B, N, D,
        X.stride(0), X.stride(1), X.stride(2),
        W_packed.stride(0), W_packed.stride(1),
        Y.stride(0), Y.stride(1), Y.stride(2),
        BLOCK_D=block_d
    )
    return Y.squeeze(1) if was_2d else Y
