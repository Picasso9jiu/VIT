import time
import random
import torch
import numpy as np
from torch.nn.parameter import Parameter

from utils import *
from quant import *
from quant.quantizer import UniformQuantizer, LogSqrt2Quantizer

# ===================== 配置 =====================
DEVICE = "cuda"
BATCH_SIZE = 32
WARMUP_ITER = 20
TEST_ITER = 100
# =================================================

def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _fast_init_quantizers(model, device):
    """跳过昂贵的 torch.quantile 搜索，直接用占位值初始化量化器"""
    for m in model.modules():
        if isinstance(m, (UniformQuantizer, LogSqrt2Quantizer)):
            if not m.inited:
                m.delta = Parameter(torch.tensor(1.0, device=device))
                if hasattr(m, "zero_point"):
                    m.zero_point = Parameter(torch.tensor(0.0, device=device))
                m.inited.fill_(1)


def load_full_model(pth_path, model_timm_name, device):
    """加载全精度模型（跳过 HF 下载，用本地权重）"""
    model = build_model(model_timm_name, pretrained=False)
    state_dict = torch.load(pth_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def load_quant_model(pth_path, model_timm_name, w_bits, a_bits, w_cw, a_cw, device):
    """加载量化模型（跳过 HF 下载，用本地权重）"""
    base_model = build_model(model_timm_name, pretrained=False)
    wq_params = {'n_bits': w_bits, 'channel_wise': w_cw}
    aq_params = {'n_bits': a_bits, 'channel_wise': a_cw}
    q_model = quant_model(base_model, input_quant_params=aq_params, weight_quant_params=wq_params)
    q_model.to(device)
    q_model.eval()

    # Stage 2 reparameterization 把 qkv/fc1 的 per-channel 合并为标量
    for block in q_model.blocks:
        block.attn.qkv.input_quantizer.channel_wise = False
        block.mlp.fc1.input_quantizer.channel_wise = False

    set_quant_state(q_model, input_quant=True, weight_quant=True)

    # 用占位值快速初始化，不走昂贵的 torch.quantile 搜索
    _fast_init_quantizers(q_model, device)

    # 加载训练好的量化权重（会覆盖占位值）
    state_dict = torch.load(pth_path, map_location=device)
    q_model.load_state_dict(state_dict)

    set_quant_state(q_model, input_quant=True, weight_quant=True)
    return q_model


def benchmark(model, x, warmup, iterations, name):
    """测速：预热 + 正式计时"""
    # 预热
    for _ in range(warmup):
        _ = model(x)
    torch.cuda.synchronize()

    # 正式计时
    t0 = time.time()
    for _ in range(iterations):
        _ = model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    fps = BATCH_SIZE * iterations / elapsed
    ms_per_batch = elapsed / iterations * 1000
    print(f"  {name:16s} | {elapsed:6.2f}s | {ms_per_batch:7.2f}ms/batch | {fps:8.1f} FPS")
    return elapsed


def main():
    seed(0)

    model_zoo = {
        'vit_base': 'vit_base_patch16_224',
    }

    timm_name = model_zoo['vit_base']

    print("=" * 70)
    print("  Loading models...")
    print("=" * 70)

    # 1. 全精度模型
    model_fp32 = load_full_model("vit_base_full_pretrained.pth", timm_name, DEVICE)

    # 2. W8A8 量化模型
    model_w8a8 = load_quant_model(
        "vit_base_w8a8.pth", timm_name,
        w_bits=8, a_bits=8, w_cw=True, a_cw=False, device=DEVICE
    )

    # 3. W4A4 量化模型
    model_w4a4 = load_quant_model(
        "vit_base_w4a4.pth", timm_name,
        w_bits=4, a_bits=4, w_cw=True, a_cw=False, device=DEVICE
    )

    print("  All models loaded successfully!")
    print()

    # 测试数据
    x = torch.randn(BATCH_SIZE, 3, 224, 224).to(DEVICE)

    print("=" * 70)
    print(f"  Benchmark: batch_size={BATCH_SIZE}, iterations={TEST_ITER}")
    print("=" * 70)
    print(f"  {'Model':16s} | {'Total':>6s} | {'Latency':>8s} | {'Throughput':>9s}")
    print("  " + "-" * 52)

    fp32_time = benchmark(model_fp32, x, WARMUP_ITER, TEST_ITER, "FP32 (full prec)")
    w8_time = benchmark(model_w8a8, x, WARMUP_ITER, TEST_ITER, "W8A8 (quantized)")
    w4_time = benchmark(model_w4a4, x, WARMUP_ITER, TEST_ITER, "W4A4 (quantized)")

    print("  " + "-" * 52)

    # 说明
    print()
    print("  NOTE: Since these are pseudo-quantized models (weights stored as FP32,")
    print("  GEMM runs in FP32), the speed will be similar across all three.")
    print("  The speed of W8A8/W4A4 may even be slightly slower due to the extra")
    print("  round/clamp operations in the quantizer forward pass.")
    print("=" * 70)

    # 打印加速比（预期接近 1.0）
    print(f"\n  W8A8  vs FP32: {fp32_time/w8_time:.2f}x")
    print(f"  W4A4  vs FP32: {fp32_time/w4_time:.2f}x")


if __name__ == "__main__":
    main()
