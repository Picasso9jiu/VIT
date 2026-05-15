"""
Real INT8/FP16 quantization via ONNX Runtime + CUDA EP.
Converts FP32 ONNX → FP16 / INT8 ONNX → benchmarks with GPU acceleration.

This gives REAL hardware deployment numbers for the paper.

Usage:
    python ex_ort_quant.py --batch_size 8 --iters 20
"""

import os, sys, time, argparse, numpy as np

# Fix cuDNN DLL path BEFORE onnxruntime import
_torch_lib = os.path.join(os.path.dirname(__file__) or ".",
                           r"d:\miniconda3\envs\ultralytics\Lib\site-packages\torch\lib")
import glob as _glob
_candidates = [
    _torch_lib,
    r"d:\miniconda3\envs\ultralytics\Lib\site-packages\torch\lib",
]
for _d in _candidates:
    if os.path.isdir(_d):
        os.environ["PATH"] = _d + ";" + os.environ.get("PATH", "")
        os.add_dll_directory(_d)
        break

import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType, CalibrationMethod
from onnxruntime.quantization.calibrate import CalibrationDataReader
import onnx
from onnxconverter_common import float16


class RandomCalibrationDataReader(CalibrationDataReader):
    """Calibration data reader using random ImageNet-like data."""
    def __init__(self, batch_size, num_batches, input_name):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.input_name = input_name
        self.iter = iter(self)

    def get_next(self):
        return next(self.iter, None)

    def __iter__(self):
        for _ in range(self.num_batches):
            yield {self.input_name: np.random.randn(
                self.batch_size, 3, 224, 224).astype(np.float32)}


def benchmark_ort(model_path, batch_size, warmup, iters):
    """Benchmark an ONNX model with CUDA EP."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        model_path, opts,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    actual_ep = session.get_providers()
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_type = input_info.type  # "tensor(float)" or "tensor(float16)"

    # Match input dtype
    if "float16" in input_type:
        x = np.random.randn(batch_size, 3, 224, 224).astype(np.float16)
    else:
        x = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        session.run(None, {input_name: x})

    # Timed
    t0 = time.time()
    for _ in range(iters):
        session.run(None, {input_name: x})
    elapsed = time.time() - t0

    fps = batch_size * iters / elapsed
    ms = elapsed / iters * 1000
    model_mb = os.path.getsize(model_path) / 1024**2
    return elapsed, fps, ms, model_mb, actual_ep[0]


def main():
    parser = argparse.ArgumentParser("ORT Real Quantization Benchmark")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    B = args.batch_size
    warmup = args.warmup
    iters = args.iters

    print("=" * 85)
    print("  ONNX Runtime Real Quantization Benchmark (CUDA EP)")
    print(f"  Batch={B}, Warmup={warmup}, Iters={iters}")
    print("=" * 85)

    fp32_onnx = "vit_base_fp32.onnx"
    fp16_onnx = "vit_base_fp16_ort.onnx"
    int8_onnx = "vit_base_int8_ort.onnx"

    assert os.path.exists(fp32_onnx), "Run: python to_onnx.py --fp32 first"

    # ---- 1. FP32 baseline ----
    print("\n[1/3] FP32 ONNX + CUDA EP")
    t, fps, ms, sz, ep = benchmark_ort(fp32_onnx, B, warmup, iters)
    print(f"  FP32:  {fps:7.1f} FPS | {ms:6.2f}ms | {sz:.0f} MB | EP: {ep}")
    fp32_fps = fps

    # ---- 2. FP16 ONNX (convert) ----
    print("\n[2/3] Converting FP32 → FP16 ONNX...")
    if not os.path.exists(fp16_onnx):
        print("  Converting...", end=" ", flush=True)
        fp32_model = onnx.load(fp32_onnx)
        fp16_model = float16.convert_float_to_float16(fp32_model)
        onnx.save(fp16_model, fp16_onnx)
        print("done")
    else:
        print("  Already exists, using cached")
    t, fps, ms, sz, ep = benchmark_ort(fp16_onnx, B, warmup, iters)
    print(f"  FP16:  {fps:7.1f} FPS | {ms:6.2f}ms | {sz:.0f} MB | EP: {ep}")

    # ---- 3. INT8 ONNX (quantize with calibration) ----
    print("\n[3/3] Quantizing FP32 → INT8 ONNX (with calibration)...")
    if not os.path.exists(int8_onnx):
        print("  Calibrating...", end=" ", flush=True)
        # Use 100 batches of calibration data
        calib_reader = RandomCalibrationDataReader(B, num_batches=100, input_name="input")
        quantize_static(
            fp32_onnx,
            int8_onnx,
            calibration_data_reader=calib_reader,
            quant_format=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.Percentile,
        )
        print("done")
    else:
        print("  Already exists, using cached")
    t, fps, ms, sz, ep = benchmark_ort(int8_onnx, B, warmup, iters)
    print(f"  INT8:  {fps:7.1f} FPS | {ms:6.2f}ms | {sz:.0f} MB | EP: {ep}")
    int8_fps = fps

    # ---- Summary ----
    print("\n" + "=" * 85)
    print("  RESULTS: Real Hardware Deployment")
    print("=" * 85)
    print(f"  {'Model':16s} | {'FPS':>8s} | {'vs FP32':>8s} | {'Size':>8s} | {'Size vs FP32':>12s} | {'Latency':>9s}")
    print("  " + "-" * 75)

    for name, model_fps, model_path in [
        ("FP32 (baseline)", fp32_fps, fp32_onnx),
        ("FP16", benchmark_ort(fp16_onnx, B, warmup, iters)[1], fp16_onnx),
        ("INT8", int8_fps, int8_onnx),
    ]:
        size_mb = os.path.getsize(model_path) / 1024**2
        fp32_size = os.path.getsize(fp32_onnx) / 1024**2
        print(f"  {name:<16s} | {model_fps:8.1f} | {model_fps/fp32_fps:7.2f}x | {size_mb:7.1f} MB | {size_mb/fp32_size:11.2f}x | {1000/model_fps:8.2f}ms")

    print(f"\n  GPU: CUDA EP (cuDNN + cuBLAS)")
    print(f"  Calibration: 100 batches of random ImageNet-like data")
    print(f"\n  For the paper: INT8 achieves {int8_fps/fp32_fps:.1f}x real speedup")
    print(f"  with {os.path.getsize(int8_onnx)/os.path.getsize(fp32_onnx):.1f}x model size reduction")
    print("  Done!")


if __name__ == "__main__":
    main()
