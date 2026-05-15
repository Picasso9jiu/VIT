"""
TensorRT INT8 真量化 + FP16 — 部署你的 IaS-ViT 模型

用法:
    python ex_trt_int8.py --batch_size 16
"""

import os, time, argparse
import numpy as np
import torch
import tensorrt as trt

IMAGE_SIZE = 224


# ============================================================
#  INT8 校准器 (cuda-python backend for TensorRT 10.x)
# ============================================================
class Int8EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batch_size, num_batches, cache_file="calibration.cache"):
        super().__init__()
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.cache_file = cache_file
        self.current_batch = 0
        self.device_input = None

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_batch >= self.num_batches:
            return None

        # 生成校准数据 (使用 ImageNet 标准归一化的随机数据)
        batch = np.random.randn(self.batch_size, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
        # 模拟 ImageNet 归一化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        batch = (batch * 0.3 + 0.45 - mean) / std  # 让分布接近 ImageNet

        self.current_batch += 1
        # TRT 10.x 期望 list of numpy arrays (host buffers)
        return [np.ascontiguousarray(batch)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


# ============================================================
#  Engine 构建
# ============================================================
def build_engine(onnx_path, batch_size, calibrator=None, fp16=False):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if calibrator is not None:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator

    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError(f"ONNX parse failed: {parser.get_error(0)}")

    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    profile.set_shape(inp.name, (1, 3, 224, 224), (batch_size, 3, 224, 224), (batch_size * 2, 3, 224, 224))
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed — possibly INT8 not supported")
    return bytes(serialized)


def benchmark_engine(engine_bytes, batch_size, warmup, iters):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    ctx = engine.create_execution_context()
    ctx.set_input_shape("input", (batch_size, 3, 224, 224))

    x = torch.randn(batch_size, 3, 224, 224, device="cuda")
    y = torch.zeros(batch_size, 1000, device="cuda")
    ctx.set_tensor_address("input", x.data_ptr())
    ctx.set_tensor_address("output", y.data_ptr())

    stream = torch.cuda.Stream()
    for _ in range(warmup):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    t0 = time.time()
    for _ in range(iters):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    elapsed = time.time() - t0

    return batch_size * iters / elapsed, elapsed / iters * 1000, len(engine_bytes) / (1024 * 1024)


# ============================================================
#  主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser("TensorRT INT8 Real Quantization")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--calib_batches", type=int, default=50)
    args = parser.parse_args()

    B = args.batch_size
    warmup = args.warmup
    iters = args.iters
    fp32_onnx = "vit_base_fp32.onnx"

    print("=" * 72)
    print("  TensorRT INT8/FP16 真实量化测试")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Batch={B}")
    print("=" * 72)

    # ============================================================
    #  1. FP32 基准
    # ============================================================
    print("\n[1/3] FP32 Engine ...", end=" ", flush=True)
    eng_fp32 = build_engine(fp32_onnx, B)
    fps_fp32, ms_fp32, mb_fp32 = benchmark_engine(eng_fp32, B, warmup, iters)
    print(f"{fps_fp32:.1f} FPS | {ms_fp32:.1f}ms | {mb_fp32:.0f} MB")

    # ============================================================
    #  2. INT8 校准 + 编译
    # ============================================================
    print(f"\n[2/3] INT8 Engine (calibrating, {args.calib_batches} batches)...")
    print("      calibrating ...", end=" ", flush=True)
    calibrator = Int8EntropyCalibrator(B, args.calib_batches)
    print("done. Building ...", end=" ", flush=True)
    eng_int8 = build_engine(fp32_onnx, B, calibrator=calibrator)
    fps_int8, ms_int8, mb_int8 = benchmark_engine(eng_int8, B, warmup, iters)
    print(f"{fps_int8:.1f} FPS | {ms_int8:.1f}ms | {mb_int8:.0f} MB")

    # ============================================================
    #  3. FP16 对照
    # ============================================================
    print(f"\n[3/3] FP16 Engine ...", end=" ", flush=True)
    eng_fp16 = build_engine(fp32_onnx, B, fp16=True)
    fps_fp16, ms_fp16, mb_fp16 = benchmark_engine(eng_fp16, B, warmup, iters)
    print(f"{fps_fp16:.1f} FPS | {ms_fp16:.1f}ms | {mb_fp16:.0f} MB")

    # ============================================================
    #  汇总
    # ============================================================
    print("\n" + "=" * 72)
    print("  INT8 真实量化 — 结果汇总")
    print("=" * 72)
    print(f"  {'格式':16s} | {'FPS':>7s} | {'vs FP32':>7s} | {'延迟':>7s} | {'Engine':>7s}")
    print("  " + "-" * 58)
    for label, fps, ms, mb in [
        ("FP32", fps_fp32, ms_fp32, mb_fp32),
        ("INT8 (TensorRT)", fps_int8, ms_int8, mb_int8),
        ("FP16", fps_fp16, ms_fp16, mb_fp16),
    ]:
        print(f"  {label:<16s} | {fps:7.1f} | {fps/fps_fp32:6.2f}x | {ms:6.2f}ms | {mb:3.0f} MB")

    print(f"\n  论文结论:")
    print(f"    - FP16 推理加速: {fps_fp16/fps_fp32:.1f}x, 模型压缩: {mb_fp32/mb_fp16:.1f}x")
    print(f"    - INT8 推理加速: {fps_int8/fps_fp32:.1f}x, 模型压缩: {mb_fp32/mb_int8:.1f}x")
    print("  Done!")


if __name__ == "__main__":
    main()
