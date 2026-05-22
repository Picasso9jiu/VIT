"""
对比 torchao 官方均匀量化 vs 我们的 PTQ 非对称量化 kernel
============================================================
纯 torchao API，不做任何自定义优化。
WSL2 运行: python test_wsl/ex_torchao_bench.py
"""
import torch, time, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from types import MethodType
from utils.build_model import build_model
from timm.models.vision_transformer import Attention

device = "cuda"


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


def bench(name, model, x, warmup, iters):
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
    print(f"  {name:<36s} | {fps:7.1f} FPS | {ms:6.2f}ms")
    return fps


def load_base():
    base = build_model("vit_base_patch16_224", pretrained=False)
    for m in base.modules():
        if isinstance(m, Attention):
            m.forward = MethodType(attn_fwd, m)
    base.load_state_dict(
        torch.load("vit_base_full_pretrained.pth", map_location="cpu"), strict=False
    )
    base.to(device).eval()
    return base


if __name__ == "__main__":
    B, W, I = 8, 5, 20
    x_fp32 = torch.randn(B, 3, 224, 224, device=device)

    with torch.no_grad():
        # ---- FP32 baseline ----
        print("[1/5] FP32 baseline...", flush=True)
        base = load_base()
        fp32 = bench("FP32 baseline", base, x_fp32, W, I)
        del base; torch.cuda.empty_cache()

        # ---- torchao INT8 weight-only (纯官方) ----
        print("[2/5] torchao INT8 weight-only...", flush=True)
        from torchao.quantization import quantize_, int8_weight_only
        base = load_base()
        quantize_(base, int8_weight_only(), filter_fn=lambda m, n: isinstance(m, torch.nn.Linear))
        tao_w8 = bench("torchao INT8 weight-only", base, x_fp32, W, I)
        del base; torch.cuda.empty_cache()

        # ---- torchao INT4 weight-only ----
        # filter_fn 是 quantize_ 官方参数，只量化 Linear 层，跳过 MatMul / LayerNorm 等
        print("[3/5] torchao INT4 weight-only...", flush=True)
        from torchao.quantization import int4_weight_only
        base = load_base()
        quantize_(base, int4_weight_only(), filter_fn=lambda m, n: isinstance(m, torch.nn.Linear))
        tao_w4 = bench("torchao INT4 weight-only", base, x_fp32, W, I)
        del base; torch.cuda.empty_cache()

        # ---- 我们的 PTQ + Triton kernel ----
        print("[4/5] Our W8A8 — PTQ non-uniform + Triton...", flush=True)
        from test_wsl.ex_benchmark import load_and_convert
        x_bf16 = torch.randn(B, 3, 224, 224, device=device, dtype=torch.bfloat16)
        q8, _ = load_and_convert("vit_base_w8a8.pth", w_bits=8)
        our_w8 = bench("Our W8A8 (non-uniform, Triton)", q8, x_bf16, W, I)
        del q8; torch.cuda.empty_cache()

        print("[5/5] Our W4A4 — PTQ non-uniform + Triton...", flush=True)
        q4, _ = load_and_convert("vit_base_w4a4.pth", w_bits=4)
        our_w4 = bench("Our W4A4 (non-uniform, Triton)", q4, x_bf16, W, I)
        del q4; torch.cuda.empty_cache()

        # ---- Summary ----
        rows = [
            ("FP32 baseline",               f"{fp32:.1f}"),
            ("torchao INT8 (uniform)",       f"{tao_w8:.1f}"),
            ("torchao INT4 (uniform)",       f"{tao_w4:.1f}" if tao_w4 else "FAILED"),
            ("Our W8A8 (PTQ + Triton)",      f"{our_w8:.1f}"),
            ("Our W4A4 (PTQ + Triton)",      f"{our_w4:.1f}"),
        ]

        print(f"\n  {'─'*56}")
        print(f"  {'Method':<31s} {'FPS':>7s}  {'vs FP32':>10s}")
        print(f"  {'─'*56}")
        for name, val in rows:
            spd = f"{float(val)/fp32:.2f}x" if val != "FAILED" else "  --"
            print(f"  {name:<31s} {val:>7s}  {spd:>10s}")
        print(f"  {'─'*56}")
        print(f"  torchao = official uniform quantization, pure API call")
        print(f"  Our     = PTQ asymmetric quantization + custom Triton kernel")
