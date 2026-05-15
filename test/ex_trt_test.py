"""
TensorRT 硬件推理测试 — 用你的 W8A8/W4A4 量化模型
=====================================================
将 ONNX 模型编译为 TensorRT Engine，在 GPU 上做真实推理测速。

用法:
    python ex_trt_test.py                          # 全部 6 组测试
    python ex_trt_test.py --batch_size 4 --iters 50
"""

import os, time, argparse, torch
import tensorrt as trt


# ============================================================
#  TensorRT Engine 构建
# ============================================================
def build_engine(onnx_path, batch_size):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # EXPLICIT_BATCH 模式支持动态 batch
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GB

    # 解析 ONNX
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError(f"ONNX parse failed: {parser.get_error(0)}")

    # 动态 batch profile
    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    profile.set_shape(
        inp.name,
        (1, 3, 224, 224),                # min
        (batch_size, 3, 224, 224),       # opt
        (batch_size * 2, 3, 224, 224),   # max
    )
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    return bytes(serialized)


# ============================================================
#  TensorRT 推理测速
# ============================================================
def benchmark_engine(engine_bytes, batch_size, warmup, iters):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    ctx = engine.create_execution_context()

    # TensorRT 10.x API: 显式绑定输入/输出地址
    ctx.set_input_shape("input", (batch_size, 3, 224, 224))
    x = torch.randn(batch_size, 3, 224, 224, device="cuda")
    y = torch.zeros(batch_size, 1000, device="cuda")
    ctx.set_tensor_address("input", x.data_ptr())
    ctx.set_tensor_address("output", y.data_ptr())

    stream = torch.cuda.Stream()

    # 预热
    for _ in range(warmup):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    # 正式计时
    t0 = time.time()
    for _ in range(iters):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    elapsed = time.time() - t0

    fps = batch_size * iters / elapsed
    ms = elapsed / iters * 1000
    engine_mb = len(engine_bytes) / (1024 * 1024)
    return fps, ms, engine_mb


# ============================================================
#  主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser("TensorRT Hardware Benchmark")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    B = args.batch_size
    warmup = args.warmup
    iters = args.iters

    # 测试矩阵 — 只测你的三个模型
    tests = [
        ("IaS-ViT  FP32", "vit_base_fp32.onnx"),
        ("IaS-ViT  W8A8", "vit_base_w8a8.onnx"),
        ("IaS-ViT  W4A4", "vit_base_w4a4.onnx"),
    ]

    print("=" * 78)
    print("  TensorRT 硬件推理测试 — 你的 IaS-ViT 量化模型")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Batch={B}, Warmup={warmup}, Iters={iters}")
    print("=" * 78)

    results = []
    for name, onnx_path in tests:
        print(f"\n  Building: {name} ...", end=" ", flush=True)
        engine_bytes = build_engine(onnx_path, B)
        print(f"{len(engine_bytes)/1024**2:.0f} MB engine ...", end=" ", flush=True)
        fps, ms, eng_mb = benchmark_engine(engine_bytes, B, warmup, iters)
        print(f"{fps:.1f} FPS", flush=True)
        results.append((name, fps, ms, eng_mb))

    # 汇总
    print("\n" + "=" * 78)
    print("  测试结果汇总")
    print("=" * 78)
    baseline_fps = results[0][1]
    print(f"  {'模型':30s} | {'FPS':>7s} | {'vs FP32':>7s} | {'延迟':>7s} | {'Engine':>7s}")
    print("  " + "-" * 68)
    for name, fps, ms, eng_mb in results:
        print(f"  {name:<30s} | {fps:7.1f} | {fps/baseline_fps:6.2f}x | {ms:6.2f}ms | {eng_mb:5.0f} MB")

    print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  TensorRT: {trt.__version__}")
    print("  Done!")


if __name__ == "__main__":
    main()
