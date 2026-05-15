"""
Real INT4/INT8 quantization with torchao for hardware acceleration testing.

Uses torchao's CUDA-optimized kernels to quantize nn.Linear layers
to real INT4/INT8 and benchmark inference speed.

Usage:
    python ex_torchAO.py --w_bits 4              # INT4 weight-only
    python ex_torchAO.py --w_bits 8              # INT8 weight-only
    python ex_torchAO.py --w_bits 8 --aq         # INT8 dynamic act + INT8 weight
    python ex_torchAO.py --w_bits 4 --group 64   # INT4 weight-only, group_size=64
"""

import time
import argparse
import torch
from utils.build_model import build_model


def load_fp32_model(pth_path, device):
    """Load FP32 ViT-Base model with MatMul-replaced attention (no pseudo-quant)."""
    print("  Building ViT-Base model architecture...")
    model = build_model("vit_base_patch16_224", pretrained=False)
    print(f"  Loading weights from {pth_path}...")
    state_dict = torch.load(pth_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print("  FP32 model loaded successfully.")
    return model


def apply_torchao_quant(model, w_bits=4, aq=False, group_size=128):
    """Apply torchao real quantization to the model in-place.

    Replaces nn.Linear layers with quantized implementations using
    CUDA-optimized kernels (cuBLAS for INT8, tinygemm for INT4).

    For INT4, the model must be bfloat16 first (torchao TensorCoreTiledLayout
    requires matching scale/zero_point dtypes).
    """
    from torchao.quantization import (
        int4_weight_only,
        int8_weight_only,
        int8_dynamic_activation_int8_weight,
        quantize_,
    )

    if w_bits == 4:
        model = model.to(dtype=torch.bfloat16)
        config = int4_weight_only(group_size=group_size)
        label = f"INT4 weight-only (group={group_size}, bf16)"
    elif w_bits == 8:
        if aq:
            config = int8_dynamic_activation_int8_weight()
            label = "INT8 weight + dynamic INT8 act"
        else:
            config = int8_weight_only()
            label = "INT8 weight-only"

    print(f"  Applying: {label}")
    quantize_(model, config)
    print("  Quantization applied successfully.")
    return model, label


def benchmark(model, x, warmup, iterations, name, device, batch_size):
    """Warmup then timed benchmark."""
    # Ensure x matches model dtype
    model_dtype = next(model.parameters()).dtype
    if x.dtype != model_dtype:
        x = x.to(dtype=model_dtype)

    # Warmup
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed run
    t0 = time.time()
    for _ in range(iterations):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    fps = batch_size * iterations / elapsed
    ms = elapsed / iterations * 1000
    print(f"  {name:24s} | {elapsed:6.2f}s | {ms:7.2f}ms/batch | {fps:8.1f} FPS")
    return elapsed


def main():
    parser = argparse.ArgumentParser("torchao Real Quantization Benchmark")
    parser.add_argument("--w_bits", type=int, default=4, choices=[4, 8],
                        help="Weight quantization bit-width (4 or 8)")
    parser.add_argument("--aq", action="store_true",
                        help="Enable dynamic INT8 activation quantization (only for --w_bits 8)")
    parser.add_argument("--group", type=int, default=128,
                        help="Group size for INT4 weight quantization (default: 128)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (default: 16, try 4 or 8 if GPU has < 6GB)")
    parser.add_argument("--iters", type=int, default=20,
                        help="Benchmark iterations (default: 20)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup iterations (default: 5)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    warmup = args.warmup
    iters = args.iters

    pth_path = "vit_base_full_pretrained.pth"

    print("=" * 70)
    print("  torchao Real Quantization Benchmark")
    print(f"  Weight bits: {args.w_bits}  |  Activation quant: {args.aq}")
    if args.w_bits == 4:
        print(f"  Group size: {args.group}")
    print(f"  Batch={batch_size}, Warmup={warmup}, Iters={iters}")
    print(f"  Device: {device}")
    print("=" * 70)

    # 1. Load model
    print("\n[1/4] Loading model...")
    model = load_fp32_model(pth_path, device)

    # 2. Benchmark FP32 baseline
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    print(f"\n[2/4] Benchmarking FP32 baseline...")
    print(f"  {'Model':24s} | {'Time':>6s} | {'Latency':>8s} | {'FPS':>8s}")
    print("  " + "-" * 60)
    fp32_time = benchmark(model, x, warmup, iters, "FP32", device, batch_size)

    # 3. If INT4, also benchmark bfloat16 baseline for fair comparison
    if args.w_bits == 4:
        print(f"\n[3/4] Converting to bfloat16 baseline...")
        model_bf16 = model.to(dtype=torch.bfloat16)
        bf16_time = benchmark(model_bf16, x, warmup, iters,
                              "BF16 (no quant)", device, batch_size)
        model = model_bf16  # continue with bf16 model for quantization

    # 4. Apply torchao quantization & benchmark
    step = 4 if args.w_bits == 4 else 3
    print(f"\n[{step}/4] Applying torchao quantization...")
    model, label = apply_torchao_quant(model, args.w_bits, args.aq, args.group)

    print("\n  Re-benchmarking quantized model...")
    quant_time = benchmark(model, x, warmup, iters, label, device, batch_size)

    print("  " + "-" * 60)

    # Summary
    print(f"\n  Results:")
    print(f"    FP32 baseline:          {fp32_time:.2f}s")
    if args.w_bits == 4:
        print(f"    BF16 baseline:          {bf16_time:.2f}s")
        print(f"    INT4  vs FP32:          {fp32_time/quant_time:.2f}x")
        print(f"    INT4  vs BF16:          {bf16_time/quant_time:.2f}x")
    else:
        print(f"    {label}:  {quant_time:.2f}s")
        print(f"    Speedup vs FP32:        {fp32_time/quant_time:.2f}x")

    if device.type == "cuda":
        mem = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"    Peak GPU memory:        {mem:.0f} MiB")

    print("\n  Done!")


if __name__ == "__main__":
    main()
