"""
Export IaS-ViT models to ONNX format.
Usage:
    python to_onnx.py --fp32                        # 导出全精度 FP32 模型
    python to_onnx.py --model vit_base_w4a4.pth --w_bits 4 --w_cw
    python to_onnx.py --model vit_base_w8a8.pth --w_bits 8 --w_cw
"""
import argparse
import os
import torch
from torch.nn.parameter import Parameter

from utils.build_model import build_model
from quant.quant_model import quant_model, set_quant_state
from quant.quantizer import UniformQuantizer, LogSqrt2Quantizer


def _init_quantizers_from_ckpt(model, ckpt, device):
    """从 checkpoint 匹配量化器形状，用占位值初始化。
    遍历 checkpoint 中所有 quantizer 的 delta/zero_point，
    按同形状创建 Parameter，然后 load_state_dict 就能成功覆盖。"""
    count = 0
    inited_ids = set()

    for ckpt_key, ckpt_val in ckpt.items():
        if 'quantizer' not in ckpt_key:
            continue
        if not (ckpt_key.endswith('.delta') or ckpt_key.endswith('.zero_point')):
            continue

        # 按点号路径导航到量化器模块
        parts = ckpt_key.split('.')
        obj = model
        for part in parts[:-1]:
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)

        param_name = parts[-1]
        setattr(obj, param_name, Parameter(torch.zeros(ckpt_val.shape, device=device)))

        oid = id(obj)
        if oid not in inited_ids:
            inited_ids.add(oid)
            obj.inited.fill_(1)
            # LogSqrt2Quantizer 在 init_quantization_scale 中设置 base/maxv/minv
            # 我们跳过了该初始化，需要手动补上，否则 forward 时 AttributeError
            if isinstance(obj, LogSqrt2Quantizer):
                obj.base = 2
                obj.maxv = 1.0
                obj.minv = -1.0
            count += 1

    # 兜底：处理 checkpoint 中不存在的量化器（理论上不应该有）
    for m in model.modules():
        if isinstance(m, (UniformQuantizer, LogSqrt2Quantizer)):
            if not m.inited:
                m.delta = Parameter(torch.tensor(1.0, device=device))
                if hasattr(m, 'zero_point'):
                    m.zero_point = Parameter(torch.tensor(0.0, device=device))
                if isinstance(m, LogSqrt2Quantizer):
                    m.base = 2
                    m.maxv = 1.0
                    m.minv = -1.0
                m.inited.fill_(1)
                count += 1

    print(f"  ✓ 初始化了 {count} 个量化器 (从 checkpoint 匹配形状)")
    return model


def load_fp32_model(device):
    """加载全精度 FP32 模型"""
    print("  [1/2] 构建 ViT-Base 模型结构...")
    model = build_model("vit_base_patch16_224", pretrained=False)
    print("  [2/2] 加载全精度权重 (vit_base_full_pretrained.pth, 346MB)...")
    model.load_state_dict(
        torch.load("vit_base_full_pretrained.pth", map_location="cpu"),
        strict=False,
    )
    model.to(device)
    model.eval()
    print("  ✓ 全精度模型加载成功")
    return model


def load_quant_model(pth_path, w_bits, w_cw, device):
    """加载量化模型（从本地 pth，不走 HF 下载）"""
    # [1/5] 构建模型结构（pretrained=False 跳过 HF 下载）
    print("  [1/5] 构建 ViT-Base 模型结构...")
    base = build_model("vit_base_patch16_224", pretrained=False)
    print("  [2/5] 加载全精度权重 (vit_base_full_pretrained.pth, 346MB)...")
    base.load_state_dict(
        torch.load("vit_base_full_pretrained.pth", map_location="cpu"),
        strict=False,
    )
    base.to(device)
    base.eval()

    # [3/5] 包装量化模型
    print(f"  [3/5] 包装为 W{w_bits}A{w_bits} 量化模型...")
    wq = {"n_bits": w_bits, "channel_wise": w_cw}
    aq = {"n_bits": w_bits, "channel_wise": False}
    q_model = quant_model(base, input_quant_params=aq, weight_quant_params=wq)
    q_model.to(device)
    q_model.eval()

    # Stage 2 会把 qkv/fc1 的 per-channel 输入量化参数合并为标量
    for block in q_model.blocks:
        block.attn.qkv.input_quantizer.channel_wise = False
        block.mlp.fc1.input_quantizer.channel_wise = False

    set_quant_state(q_model, input_quant=True, weight_quant=True)

    # [4/5] 先读 checkpoint，按实际形状初始化量化器，再 load_state_dict
    print(f"  [4/5] 加载量化权重 ({pth_path}, ~347MB)...")
    ckpt = torch.load(pth_path, map_location=device)
    _init_quantizers_from_ckpt(q_model, ckpt, device)
    q_model.load_state_dict(ckpt)
    print("  ✓ 量化模型加载成功")
    return q_model


def export_onnx(model, onnx_path, device):
    """导出 ONNX"""
    print(f"  [5/5] 导出 ONNX → {onnx_path} (ViT-Base 约需 1-3 分钟)...")
    dummy = torch.randn(1, 3, 224, 224).to(device)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"  ✓ ONNX 导出成功 → {onnx_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser("IaS-ViT → ONNX Export")
    parser.add_argument("--fp32", action="store_true",
                        help="导出全精度 FP32 模型 (不需要 --model/--w_bits/--w_cw)")
    parser.add_argument("--model", default=None,
                        help="量化 pth 文件名, e.g. vit_base_w4a4.pth")
    parser.add_argument("--w_bits", type=int, default=4,
                        help="权重量化位数 (4 或 8)")
    parser.add_argument("--w_cw", action="store_true",
                        help="权重逐通道量化 (需与训练时一致)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    assert os.path.exists("vit_base_full_pretrained.pth"), \
        "找不到 vit_base_full_pretrained.pth，请先运行 model_download.py"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.fp32:
        onnx_path = "vit_base_fp32.onnx"
        print("=" * 60)
        print(f"  IaS-ViT → ONNX 导出 (FP32 全精度)")
        print(f"  输出文件 : {onnx_path}")
        print("=" * 60)
        model = load_fp32_model(device)
    else:
        assert args.model and args.model.endswith(".pth"), "--model 需要是 .pth 文件"
        assert os.path.exists(args.model), f"找不到文件: {args.model}"
        onnx_path = args.model.replace(".pth", ".onnx")
        print("=" * 60)
        print(f"  IaS-ViT → ONNX 导出")
        print(f"  量化模型 : {args.model}")
        print(f"  量化位数 : W{args.w_bits}A{args.w_bits}")
        print(f"  输出文件 : {onnx_path}")
        print("=" * 60)
        model = load_quant_model(args.model, args.w_bits, args.w_cw, device)

    export_onnx(model, onnx_path, device)


if __name__ == "__main__":
    main()
