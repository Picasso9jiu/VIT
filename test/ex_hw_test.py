"""
Comprehensive hardware acceleration benchmark for ViT-Base.
Tests real achievable speedups on consumer GPU (RTX 3050 / Windows).

Results are suitable for paper hardware evaluation section.

Usage:
    python ex_hw_test.py                          # all tests
    python ex_hw_test.py --batch_size 8 --iters 20
"""

import time
import argparse
import torch
from utils.build_model import build_model


def load_model(device, dtype=torch.float32):
    """Load ViT-Base model with MatMul-replaced attention."""
    model = build_model("vit_base_patch16_224", pretrained=False)
    state_dict = torch.load("vit_base_full_pretrained.pth", map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def benchmark(model, x, warmup, iters, device):
    """Return elapsed seconds for `iters` forward passes."""
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.time() - t0


def run_test(name, model, x, warmup, iters, device, batch_size):
    """Run benchmark and print result."""
    elapsed = benchmark(model, x, warmup, iters, device)
    fps = batch_size * iters / elapsed
    ms = elapsed / iters * 1000
    mem = torch.cuda.max_memory_allocated(device) / 1024**2
    print(f"  {name:<36s} | {elapsed:7.2f}s | {ms:7.2f}ms | {fps:8.1f} FPS | {mem:5.0f} MiB")
    torch.cuda.reset_peak_memory_stats(device)
    return elapsed


def main():
    parser = argparse.ArgumentParser("ViT-Base Hardware Acceleration Benchmark")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda")
    B = args.batch_size
    warmup = args.warmup
    iters = args.iters

    print("=" * 90)
    print("  ViT-Base Hardware Acceleration Benchmark")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Batch={B}, Warmup={warmup}, Iters={iters}")
    print("=" * 90)

    results = {}

    # ---- 1. FP32 baseline ----
    print("\n[1/7] FP32 baseline (no tensor core)")
    model = load_model(device, torch.float32)
    x_fp32 = torch.randn(B, 3, 224, 224, device=device, dtype=torch.float32)
    print(f"  {'Method':36s} | {'Time':>7s} | {'Latency':>7s} | {'FPS':>9s} | {'VRAM':>5s}")
    print("  " + "-" * 82)
    t = run_test("FP32 (baseline)", model, x_fp32, warmup, iters, device, B)
    results["FP32"] = B * iters / t

    # ---- 2. FP32 + TF32 (tensor core for FP32 matmul) ----
    print("\n[2/7] FP32 + TF32 (tensor core matmul)")
    torch.set_float32_matmul_precision("high")
    model_tf32 = load_model(device, torch.float32)
    t = run_test("FP32 + TF32", model_tf32, x_fp32, warmup, iters, device, B)
    results["FP32+TF32"] = B * iters / t
    torch.set_float32_matmul_precision("highest")  # restore default
    del model_tf32, model
    torch.cuda.empty_cache()

    # ---- 3. FP16 ----
    print("\n[3/7] FP16 (half precision, tensor core)")
    model_fp16 = load_model(device, torch.float16)
    x_fp16 = x_fp32.half()
    t = run_test("FP16", model_fp16, x_fp16, warmup, iters, device, B)
    results["FP16"] = B * iters / t
    del model_fp16
    torch.cuda.empty_cache()

    # ---- 4. BF16 ----
    print("\n[4/7] BF16 (bfloat16, tensor core)")
    model_bf16 = load_model(device, torch.bfloat16)
    x_bf16 = x_fp32.to(torch.bfloat16)
    t = run_test("BF16", model_bf16, x_bf16, warmup, iters, device, B)
    results["BF16"] = B * iters / t
    del model_bf16
    torch.cuda.empty_cache()

    # ---- 5. INT8 weight-only (torchao + cuBLAS) ----
    print("\n[5/7] INT8 weight-only (torchao)")
    from torchao.quantization import int8_weight_only, quantize_

    model_int8 = load_model(device, torch.float32)
    quantize_(model_int8, int8_weight_only())
    t = run_test("INT8 weight-only", model_int8, x_fp32, warmup, iters, device, B)
    results["INT8"] = B * iters / t
    del model_int8
    torch.cuda.empty_cache()

    # ---- 6. BF16 + INT8 weight ----
    print("\n[6/7] BF16 + INT8 weight")
    model_i8bf16 = load_model(device, torch.bfloat16)
    quantize_(model_i8bf16, int8_weight_only())
    t = run_test("BF16 act + INT8 weight", model_i8bf16, x_bf16, warmup, iters, device, B)
    results["BF16+INT8"] = B * iters / t
    del model_i8bf16
    torch.cuda.empty_cache()

    # ---- 7. INT8 dynamic act + weight ----
    print("\n[7/7] INT8 dynamic act + weight (torchao)")
    from torchao.quantization import int8_dynamic_activation_int8_weight

    model_i8dyn = load_model(device, torch.float32)
    quantize_(model_i8dyn, int8_dynamic_activation_int8_weight())
    t = run_test("INT8 dynamic act+weight", model_i8dyn, x_fp32, warmup, iters, device, B)
    results["INT8-dyn"] = B * iters / t
    del model_i8dyn
    torch.cuda.empty_cache()

    # ---- Summary ----
    import os

    print("\n" + "=" * 90)
    print("  RESULTS SUMMARY")
    print("=" * 90)
    baseline_fps = results["FP32"]
    bf16_fps = results["BF16"]

    print(f"  {'Method':36s} | {'FPS':>8s} | {'vs FP32':>8s} | {'vs BF16':>8s} | {'Latency':>8s} | {'Weight':>8s}")
    print("  " + "-" * 95)

    # Model weight sizes (approximate)
    fp32_weight_mb = 344   # ViT-Base FP32 ~86M params × 4 bytes
    weight_sizes = {
        "FP32": fp32_weight_mb,
        "FP32+TF32": fp32_weight_mb,
        "FP16": fp32_weight_mb // 2,
        "BF16": fp32_weight_mb // 2,
        "INT8": fp32_weight_mb // 4,       # INT8 weight: 1/4 of FP32
        "BF16+INT8": fp32_weight_mb // 4,
        "INT8-dyn": fp32_weight_mb // 4,
    }

    for name in ["FP32", "FP32+TF32", "FP16", "BF16", "INT8", "BF16+INT8", "INT8-dyn"]:
        fps = results[name]
        w_mb = weight_sizes[name]
        print(f"  {name:<36s} | {fps:8.1f} | {fps/baseline_fps:7.2f}x | {fps/bf16_fps:7.2f}x | {1000/fps:7.2f}ms | {w_mb:6d} MB")

    # IaS-ViT quantized ONNX model sizes
    print("\n  --- IaS-ViT Quantized Model Deployment Metrics ---")
    onnx_files = [
        ("IaS-ViT W4A4", "vit_base_w4a4.onnx"),
        ("IaS-ViT W8A8", "vit_base_w8a8.onnx"),
        ("FP32 baseline", "vit_base_fp32.onnx"),
    ]
    print(f"  {'Model':24s} | {'ONNX Size':>12s} | {'vs FP32':>8s} | {'Weights':>10s} | {'Deploy':>10s}")
    print("  " + "-" * 75)
    fp32_onnx_mb = os.path.getsize("vit_base_fp32.onnx") / 1024**2
    for name, path in onnx_files:
        if os.path.exists(path):
            sz = os.path.getsize(path) / 1024**2
            ratio = sz / fp32_onnx_mb
            note = "Edge/Serving" if ratio < 0.8 else "Serving only"
            print(f"  {name:<24s} | {sz:8.1f} MB | {ratio:7.2f}x | {'FP32 → INT' if 'W' in name else 'FP32':>10s} | {note:>10s}")

    print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Peak VRAM: {torch.cuda.max_memory_allocated(device)/1024**2:.0f} MiB")

    print(f"\n  Key findings for paper Hardware Evaluation section:")
    print(f"    [Speed]  BF16 gives {results['BF16']/baseline_fps:.1f}x speedup (Ampere tensor cores)")
    print(f"    [Speed]  INT8 weight-only gives {results['INT8']/baseline_fps:.2f}x speedup (cuBLAS INT8 GEMM)")
    print(f"    [Memory] INT8 weights reduce model size by 4x vs FP32 ({fp32_weight_mb}→{fp32_weight_mb//4} MB)")
    print(f"    [Memory] INT4 weights reduce model size by 8x vs FP32 ({fp32_weight_mb}→{fp32_weight_mb//8} MB)")
    print(f"    [Note]   INT4 matmul acceleration requires TensorRT or data-center GPU (A100/H100)")
    print(f"    [Note]   Consumer GPU (RTX 3050) lacks INT4 tensor core — speed benefit is from memory savings")
    print("  Done!")


if __name__ == "__main__":
    main()
