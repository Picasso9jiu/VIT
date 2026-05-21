"""
精度验证：对比伪量化 W8A8/W4A4 vs Triton 部署模型（同一组权重）
"""
import torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from types import MethodType
from torch.nn.parameter import Parameter
from utils.build_model import build_model
from timm.models.vision_transformer import Attention
from quant.quant_model import quant_model, set_quant_state
from quant.quantizer import LogSqrt2Quantizer
from quant.quant_modules import QuantLinear, QuantConv2d, QuantMatMul
from test_wsl.ex_triton_kernel import AsymmInt8Linear

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


def load_pseudo(pth_path, w_bits):
    """Load pseudo-quantized model, disable input quant + Conv2d quant."""
    quant_max = 255 if w_bits == 8 else 15

    base = build_model("vit_base_patch16_224", pretrained=False)
    for m in base.modules():
        if isinstance(m, Attention):
            m.forward = MethodType(attn_fwd, m)
    base.load_state_dict(
        torch.load("vit_base_full_pretrained.pth", map_location="cpu"), strict=False)
    base.to(device).eval()

    qmodel = quant_model(base,
        {"n_bits": w_bits, "channel_wise": False},
        {"n_bits": w_bits, "channel_wise": True})
    for b in qmodel.blocks:
        b.attn.qkv.input_quantizer.channel_wise = False
        b.mlp.fc1.input_quantizer.channel_wise = False
    set_quant_state(qmodel, True, True)

    ckpt = torch.load(pth_path, map_location=device)
    for ckpt_k, ckpt_v in ckpt.items():
        if "quantizer" not in ckpt_k: continue
        parts = ckpt_k.split(".")
        obj = qmodel
        for p in parts[:-1]:
            if p.isdigit(): obj = obj[int(p)]
            else: obj = getattr(obj, p)
        setattr(obj, parts[-1], Parameter(torch.zeros(ckpt_v.shape, device=device)))
        if not obj.inited:
            obj.inited.fill_(1)
            if isinstance(obj, LogSqrt2Quantizer):
                obj.base = 2; obj.maxv = 1.0; obj.minv = -1.0
    qmodel.load_state_dict(ckpt)
    qmodel.eval()

    # Disable input quant and Conv2d quant (same as Triton path)
    for m in qmodel.modules():
        if isinstance(m, QuantConv2d): m.use_weight_quant = False
        if isinstance(m, QuantMatMul): m.use_input_quant = False
        if isinstance(m, QuantLinear): m.use_input_quant = False

    return qmodel


def convert_to_triton(qmodel, w_bits):
    """In-place: replace QuantLinear with AsymmInt8Linear, using same weights."""
    quant_max = 255 if w_bits == 8 else 15
    converted = 0
    for pn, p in qmodel.named_modules():
        for cn, c in list(p.named_children()):
            if isinstance(c, QuantLinear):
                d = c.weight_quantizer.delta.data
                zp = c.weight_quantizer.zero_point.data
                w = c.weight.data
                w_int = (w / d).round() + zp
                w_int = w_int.round().clamp(0, quant_max).to(torch.uint8)
                if d.ndim > 0:
                    s = d.view(-1); z = zp.view(-1).to(torch.uint8)
                else:
                    s = d.view(1); z = zp.view(1).to(torch.uint8)
                nl = AsymmInt8Linear(w_int, s, z, bias=c.bias)
                setattr(p, cn, nl)
                converted += 1
    qmodel = qmodel.to(dtype=torch.bfloat16)
    for m in qmodel.modules():
        if isinstance(m, AsymmInt8Linear):
            m.scale = m.scale.to(torch.float32)
    return qmodel, converted


if __name__ == "__main__":
    B = 16
    with torch.no_grad():
        x = torch.randn(B, 3, 224, 224, device=device)

        for label, pth, wb in [("W8A8", "vit_base_w8a8.pth", 8),
                               ("W4A4", "vit_base_w4a4.pth", 4)]:
            print(f"\n{'='*50}")
            print(f"  {label} — Accuracy Verification")
            print(f"{'='*50}")

            # Load pseudo-quantized (reference)
            pseudo = load_pseudo(pth, wb)
            out_p = pseudo(x)

            # Convert to Triton in-place
            triton, c = convert_to_triton(pseudo, wb)
            print(f"  Converted {c} layers to Triton kernel")
            out_t = triton(x.to(torch.bfloat16)).float()

            diff = (out_p - out_t).abs()
            agree = (out_p.argmax(dim=-1) == out_t.argmax(dim=-1)).float().mean()
            top5_p = out_p.topk(5, dim=-1).indices
            top5_agree = sum(1 for i in range(B)
                            if out_t.argmax(dim=-1)[i] in top5_p[i]) / B

            print(f"  Pseudo vs Triton (same weights, same math):")
            print(f"    Max  diff : {diff.max():.4f}")
            print(f"    Mean diff : {diff.mean():.4f}")
            print(f"    Top-1     : {agree:.4f} ({agree*100:.1f}%)")
            print(f"    Top-5     : {top5_agree:.4f} ({top5_agree*100:.1f}%)")
            print(f"    Result    : {'PASS ✓' if agree > 0.95 else 'INVESTIGATE'}")

            del pseudo
            torch.cuda.empty_cache()
