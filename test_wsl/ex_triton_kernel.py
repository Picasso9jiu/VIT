"""
Custom Triton INT8 matmul kernel using YOUR learned delta/zero_point.

Converts asymmetric INT8 weights (zp != 0) to accelerated inference:
  output = input @ ((weight_uint8 - zero_point) * delta))

Usage (in WSL2):
  python test_wsl/ex_triton_kernel.py
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _asymm_int8_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    scale_ptr, zp_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute C = A @ ((B - zp) * scale)

    A: BF16/FP16 activation  [M, K]
    B: UINT8 weight           [K, N]
    scale: FP32 per-channel   [N]
    zp:    UINT8 per-channel  [N]
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Load scale and zp for this block's N tiles
    scale = tl.load(scale_ptr + offs_n)        # [BLOCK_N]
    zp_val = tl.load(zp_ptr + offs_n).to(tl.float32)  # [BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)                          # [BLOCK_M, BLOCK_K] BF16
        b_u8 = tl.load(b_ptrs)                       # [BLOCK_K, BLOCK_N] UINT8
        # Dequantize in fp32, cast both to fp32 for dot to preserve 4-bit precision
        b_fp32 = (b_u8.to(tl.float32) - zp_val[None, :]) * scale[None, :]
        a_fp32 = a.to(tl.float32)
        acc = tl.dot(a_fp32, b_fp32, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store as float32 (Triton handles conversion to output buffer dtype)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask_m = offs_cm < M
    mask_n = offs_cn < N
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def asymm_int8_matmul(a, b_uint8, scale, zp):
    """Fused asymmetric INT8 matmul: a @ ((b_uint8 - zp) * scale)

    Args:
        a:      activation [M, K] bf16/fp16
        b_uint8: weight     [K, N] uint8
        scale:  per-channel [N] fp32
        zp:     per-channel [N] uint8
    Returns:
        [M, N] in a.dtype
    """
    M, K = a.shape
    K2, N = b_uint8.shape
    assert K == K2, f"Shape mismatch: a={a.shape}, b={b_uint8.shape}"

    c = torch.empty(M, N, dtype=a.dtype, device=a.device)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )

    _asymm_int8_gemm_kernel[grid](
        a, b_uint8, c, scale, zp,
        M, N, K,
        a.stride(0), a.stride(1),
        b_uint8.stride(0), b_uint8.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


class AsymmInt8Linear(torch.nn.Module):
    """Linear layer with YOUR asymmetric INT8/INT4 weights and Triton kernel."""

    def __init__(self, weight_uint8, scale, zp, bias=None, packed_4bit=False):
        super().__init__()
        self.register_buffer("weight", weight_uint8.contiguous().t().clone())
        # Keep scale in FP32 — bf16 loses small W4A4 deltas
        self.register_buffer("scale", scale.contiguous().to(torch.float32).clone())
        self.register_buffer("zp", zp.contiguous().to(torch.uint8))
        self.register_buffer("bias", bias.contiguous() if bias is not None
                             else torch.zeros(weight_uint8.shape[1]))
        self._packed_4bit = packed_4bit

    def forward(self, x):
        if x.dim() == 3:
            B, N, C = x.shape
            x = x.reshape(B * N, C)
            out = asymm_int8_matmul(x, self.weight, self.scale, self.zp)
            out = out.reshape(B, N, -1)
        else:
            out = asymm_int8_matmul(x, self.weight, self.scale, self.zp)
        return out + self.bias


def benchmark_model(name, model, x, warmup, iters):
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    fps = x.shape[0] * iters / elapsed
    ms = elapsed / iters * 1000
    print(f"  {name:<24s} | {fps:7.1f} FPS | {ms:6.2f}ms")
    return fps


if __name__ == "__main__":
    import os, sys, time
    from types import MethodType
    from torch.nn.parameter import Parameter
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from utils.build_model import build_model
    from timm.models.vision_transformer import Attention
    from quant.quant_model import quant_model, set_quant_state
    from quant.quantizer import LogSqrt2Quantizer
    from quant.quant_modules import QuantLinear

    def attn_fwd(s, x, attn_mask=None, is_causal=False):
        B, N, C = x.shape
        qkv = s.qkv(x).reshape(B, N, 3, s.num_heads, C // s.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        a = s.matmul1(q, k.transpose(-2, -1)) * s.scale
        a = a.softmax(dim=-1)
        a = s.attn_drop(a)
        x = s.matmul2(a, v).transpose(1, 2).reshape(B, N, C)
        x = s.proj(x)
        x = s.proj_drop(x)
        return x

    def load_and_convert(pth_path, w_bits):
        """Load your quantized model, replace QuantLinear with Triton INT4/8 kernel."""
        quant_max = 255 if w_bits == 8 else 15

        base = build_model("vit_base_patch16_224", pretrained=False)
        for m in base.modules():
            if isinstance(m, Attention):
                m.forward = MethodType(attn_fwd, m)
        base.load_state_dict(
            torch.load("vit_base_full_pretrained.pth", map_location="cpu"), strict=False
        )
        base.to(device).eval()

        qmodel = quant_model(
            base,
            {"n_bits": w_bits, "channel_wise": False},
            {"n_bits": w_bits, "channel_wise": True},
        )
        for b in qmodel.blocks:
            b.attn.qkv.input_quantizer.channel_wise = False
            b.mlp.fc1.input_quantizer.channel_wise = False
        set_quant_state(qmodel, True, True)

        ckpt = torch.load(pth_path, map_location=device)
        for ckpt_k, ckpt_v in ckpt.items():
            if "quantizer" not in ckpt_k:
                continue
            parts = ckpt_k.split(".")
            obj = qmodel
            for p in parts[:-1]:
                if p.isdigit():
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p)
            setattr(obj, parts[-1], Parameter(torch.zeros(ckpt_v.shape, device=device)))
            if not obj.inited:
                obj.inited.fill_(1)
                if isinstance(obj, LogSqrt2Quantizer):
                    obj.base = 2
                    obj.maxv = 1.0
                    obj.minv = -1.0
        qmodel.load_state_dict(ckpt)

        # Replace QuantLinear with Triton-kernel
        converted = 0
        for parent_name, parent in qmodel.named_modules():
            for child_name, child in list(parent.named_children()):
                if isinstance(child, QuantLinear):
                    d = child.weight_quantizer.delta.data
                    zp = child.weight_quantizer.zero_point.data
                    w = child.weight.data

                    w_int = (w / d + zp).round().clamp(0, quant_max).to(torch.uint8)

                    if d.ndim > 0:
                        scale = d.view(-1)
                        zp_q = zp.view(-1).to(torch.uint8)
                    else:
                        scale = d.view(1)
                        zp_q = zp.view(1).to(torch.uint8)

                    new_lin = AsymmInt8Linear(
                        w_int, scale, zp_q,
                        bias=child.bias.data if child.bias is not None else None,
                    )
                    setattr(parent, child_name, new_lin)
                    converted += 1

        qmodel = qmodel.to(dtype=torch.bfloat16)
        return qmodel, converted

    device = "cuda"
    B, W, I = 8, 5, 20

    with torch.no_grad():
        # 1. FP32 baseline
        print("[1] FP32 baseline...", flush=True)
        base = build_model("vit_base_patch16_224", pretrained=False)
        for m in base.modules():
            if isinstance(m, Attention):
                m.forward = MethodType(attn_fwd, m)
        base.load_state_dict(
            torch.load("vit_base_full_pretrained.pth", map_location="cpu"), strict=False
        )
        base.to(device).eval()
        x = torch.randn(B, 3, 224, 224, device=device)
        fp32_fps = benchmark_model("FP32", base, x, W, I)
        del base
        torch.cuda.empty_cache()

        # 2. Your W8A8 + Triton
        print("[2] Your W8A8 + Triton kernel...", flush=True)
        qmodel, c = load_and_convert("vit_base_w8a8.pth", w_bits=8)
        print(f"    Converted {c} layers")
        w8_fps = benchmark_model("Your W8A8 + Triton", qmodel, x.to(torch.bfloat16), W, I)
        del qmodel
        torch.cuda.empty_cache()

        # 3. Your W4A4 + Triton
        print("[3] Your W4A4 + Triton kernel...", flush=True)
        qmodel, c = load_and_convert("vit_base_w4a4.pth", w_bits=4)
        print(f"    Converted {c} layers")
        w4_fps = benchmark_model("Your W4A4 + Triton", qmodel, x.to(torch.bfloat16), W, I)

        print(f"\n{'='*50}")
        print(f"  Results Summary")
        print(f"{'='*50}")
        print(f"  FP32:               {fp32_fps:.1f} FPS (1.00x)")
        print(f"  Your W8A8 + Triton: {w8_fps:.1f} FPS ({w8_fps/fp32_fps:.2f}x)")
        print(f"  Your W4A4 + Triton: {w4_fps:.1f} FPS ({w4_fps/fp32_fps:.2f}x)")
