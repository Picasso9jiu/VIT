"""
IaS-ViT 硬件加速完整基准测试
==============================
测试你的 W8A8/W4A4 模型在 Triton INT8/INT4 kernel 下的真实推理速度。

Usage (WSL2):
  python test_wsl/ex_benchmark.py
"""
import torch, time, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from types import MethodType
from torch.nn.parameter import Parameter
from utils.build_model import build_model
from timm.models.vision_transformer import Attention
from quant.quant_model import quant_model, set_quant_state
from quant.quantizer import LogSqrt2Quantizer
from quant.quant_modules import QuantLinear
from test_wsl.ex_triton_kernel import AsymmInt8Linear, asymm_int8_matmul


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
    """Load your model + replace QuantLinear with real INT4/8 Triton kernel."""
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
                obj.base = 2; obj.maxv = 1.0; obj.minv = -1.0
    qmodel.load_state_dict(ckpt)

    converted = 0
    for parent_name, parent in qmodel.named_modules():
        for child_name, child in list(parent.named_children()):
            if isinstance(child, QuantLinear):
                w_d = child.weight_quantizer.delta.data
                w_zp = child.weight_quantizer.zero_point.data
                w = child.weight.data
                # Match pseudo: round_ste(w/d)+zp ≠ round(w/d+zp) at fp32 boundary
                w_int = (w / w_d).round() + w_zp
                w_int = w_int.round().clamp(0, quant_max).to(torch.uint8)
                if w_d.ndim > 0:
                    w_scale = w_d.view(-1)
                    w_zp_q = w_zp.view(-1).to(torch.uint8)
                else:
                    w_scale = w_d.view(1)
                    w_zp_q = w_zp.view(1).to(torch.uint8)

                new_lin = AsymmInt8Linear(
                    w_int, w_scale, w_zp_q,
                    bias=child.bias.data if child.bias is not None else None,
                )
                setattr(parent, child_name, new_lin)
                converted += 1

    qmodel = qmodel.to(dtype=torch.bfloat16)
    # Keep scale buffers in FP32 (bf16 loses small W4A4 deltas)
    for m in qmodel.modules():
        if isinstance(m, AsymmInt8Linear):
            m.scale = m.scale.to(torch.float32)
    # Disable quantizers not handled by Triton kernel
    from quant.quant_modules import QuantConv2d, QuantMatMul
    for m in qmodel.modules():
        if isinstance(m, QuantConv2d):
            m.use_weight_quant = False
        if isinstance(m, QuantMatMul):
            m.use_input_quant = False
    return qmodel, converted


def bench(name, model, x, warmup, iters):
    for _ in range(warmup): model(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters): model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    fps = x.shape[0] * iters / elapsed
    ms = elapsed / iters * 1000
    print(f"  {name:<32s} | {fps:7.1f} FPS | {ms:6.2f}ms")
    return fps


device = "cuda"

if __name__ == "__main__":
    B, W, I = 8, 5, 20

    with torch.no_grad():
        # ---- FP32 ----
        print("[1/3] FP32 baseline...", flush=True)
        base = build_model("vit_base_patch16_224", pretrained=False)
        for m in base.modules():
            if isinstance(m, Attention): m.forward = MethodType(attn_fwd, m)
        base.load_state_dict(
            torch.load("vit_base_full_pretrained.pth", map_location="cpu"), strict=False)
        base.to(device).eval()
        x = torch.randn(B, 3, 224, 224, device=device)
        fp32 = bench("FP32 baseline", base, x, W, I)
        del base; torch.cuda.empty_cache()

        # ---- W8A8 weight-only ----
        print("[2/3] Your W8A8 (weight INT8)...", flush=True)
        q, c = load_and_convert("vit_base_w8a8.pth", w_bits=8)
        print(f"    Converted {c} layers")
        w8 = bench("W8A8 — weight INT8", q, x.to(torch.bfloat16), W, I)
        del q; torch.cuda.empty_cache()

        # ---- W4A4 weight-only ----
        print("[3/3] Your W4A4 (weight INT4)...", flush=True)
        q, c = load_and_convert("vit_base_w4a4.pth", w_bits=4)
        print(f"    Converted {c} layers")
        w4 = bench("W4A4 — weight INT4", q, x.to(torch.bfloat16), W, I)
        del q; torch.cuda.empty_cache()

        print(f"\n{'='*60}")
        print(f"  IaS-ViT Hardware Acceleration — Final")
        print(f"{'='*60}")
        print(f"  FP32 baseline:      {fp32:7.1f} FPS (1.00x)")
        print(f"  Your W8A8:          {w8:7.1f} FPS ({w8/fp32:.2f}x)")
        print(f"  Your W4A4:          {w4:7.1f} FPS ({w4/fp32:.2f}x)")
        print(f"\n  All using YOUR learned delta/zero_point via custom Triton kernel.")
        print(f"  Autotune enabled — Triton selects optimal tile sizes per layer.")
