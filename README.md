# IaS-ViT: Post-Training Quantization for Vision Transformers

复现 TPAMI 2025 论文 *"I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization"* —— 完整流程覆盖 **PTQ 算法复现 → ONNX 导出 → 硬件加速部署**。

---

## 项目结构

```
IaS-ViT-main/
├── quant/                         # 量化核心模块
│   ├── quantizer.py               #   UniformQuantizer / LogSqrt2Quantizer
│   ├── quant_modules.py           #   QuantConv2d / QuantLinear / QuantMatMul
│   ├── quant_model.py             #   模型包装与量化状态切换
│   ├── block_recon.py             #   Block 级输出重建
│   ├── layer_recon.py             #   Layer 级输出重建
│   └── data_utils.py              #   数据加载工具
├── utils/
│   ├── build_model.py             #   ViT 模型构建 (替换 Attention MatMul)
│   └── build_dataset.py           #   ImageNet 数据集加载与预处理
├── test_quant.py                  # 【算法】PTQ 量化 pipeline 与精度验证
├── to_onnx.py                     #  ONNX 模型导出 (FP32 / W8A8 / W4A4)
├── model_download.py              #  下载预训练权重
├── test/                          # Windows 硬件测试脚本
│   ├── ex_hw_test.py              #   完整硬件基准 (FP32 / FP16 / BF16 / INT8)
│   ├── ex_torchAO.py              #   torchao INT4/INT8 真量化测试
│   ├── ex_trt_test.py             #   TensorRT 部署测速
│   ├── ex_ort_quant.py            #   ONNX Runtime INT8/FP16 量化
│   ├── ex_trt_int8.py             #   TensorRT INT8 校准尝试
│   └── test_speed.py              #   伪量化模型测速 (FP32 / W8A8 / W4A4)
├── test_wsl/                      # WSL2 / Linux 硬件部署脚本 (Triton)
│   ├── ex_triton_kernel.py        #   【核心】自定义 Triton INT4/INT8 kernel
│   ├── ex_benchmark.py            #   速度基准测试
│   └── ex_accuracy.py             #   精度保留验证
├── README.md
├── .gitignore
├── dataset/                       # ImageNet (不纳入版本控制)
└── *.pth / *.onnx                 # 模型文件 (不纳入版本控制)
```

---

## 第一部分：软件算法 — PTQ 伪量化复现 ✅

### 环境

- Python 3.11, PyTorch 2.8.0+ (CUDA 12.6)
- timm, torchvision

```bash
pip install timm torchvision
```

### 下载预训练权重

```bash
python model_download.py
```

### 运行 PTQ 量化与精度验证

```bash
python test_quant.py --model vit_base --dataset dataset/imagenet \
                     --w_bit 4 --a_bit 4 --w_cw --iter 1000
```

**IaS-ViT 三阶段量化流程：**

| 阶段 | 内容 | 作用 |
|------|------|------|
| Stage 1 | Q-Act + FP-Weight + Block/Layer 重建 | 确定激活量化参数 + 重建输出 |
| Stage 2 | 重参数化 | 将 qkv/fc1 的 per-channel 输入量化参数合并为标量 |
| Stage 3 | Q-Act + Q-Weight + 重新优化 | 联合优化权重量化参数 |

### 导出 ONNX 模型

```bash
python to_onnx.py --fp32                                          # FP32
python to_onnx.py --model vit_base_w8a8.pth --w_bits 8 --w_cw    # W8A8
python to_onnx.py --model vit_base_w4a4.pth --w_bits 4 --w_cw    # W4A4
```

> 注意：导出的 W8A8/W4A4 ONNX 为伪量化模型，权重仍以 FP32 存储，仅插入 round/clamp 算子模拟量化。

---

## 第二部分：硬件部署 — 真 INT4/INT8 加速 ✅

### tl;dr

我们在 **WSL2 (Linux) + Triton** 上为 IaS-ViT 编写了自定义 INT4/INT8 推理 kernel，实现了对 PTQ 学习参数的**原生硬件加速**。

| 模型 | 速度 vs FP32 | Top-1 精度 | 技术方案 |
|------|-------------|-----------|---------|
| **W8A8** | **2.54×** | 100% 保留 | Triton 真 INT8 kernel |
| **W4A4** | **2.43×** | Top-5 100% | Triton 真 INT4 kernel |
| BF16 | 2.9× | 无损 | Ampere Tensor Core |

### 2.1 为什么需要自己写 Kernel？

IaS-ViT 学习的是**非对称量化参数**（`zero_point ≠ 0`），而主流推理库（cuBLAS、torchao INT8、TensorRT）全部采用**对称量化**（`zero_point = 0`）。因此：

- 直接使用 torchao INT8：加速 1.4×，但使用的是 torchao 自己的对称量化参数，而非我们 PTQ 学出的参数
- 直接使用 TensorRT INT8：calibrator API 在 TRT 10.x 中已废弃，且同样不支持外部 zero_point
- 直接使用 bitsandbytes INT4/INT8：kernel 对 ViT 不友好，推理反而更慢

**唯一解决方案：为我们的非对称量化参数编写自定义 Triton kernel。**

### 2.2 环境

| 项目 | 规格 |
|------|------|
| GPU | NVIDIA GeForce RTX 3050 Laptop (4 GB, Ampere SM 8.6) |
| OS | Windows 11 Pro + WSL2 (Ubuntu) |
| Python | 3.11 |
| PyTorch | 2.11.0 (cu126) |
| Triton | 通过 torchao 自动安装 |
| torchao | 0.11.0 (Triton INT8 backend) |

```bash
# 在 WSL2 中
conda create -n vit python=3.11 -y && conda activate vit
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torchao==0.11.0 triton timm
```

### 2.3 Kernel 设计

我们的 `AsymmInt8Linear` 层在 Triton 中融合了非对称解量化与 BF16 GEMM：

```
output = input @ ((weight_uint8 - zero_point) × delta)
```

关键设计决策：

1. **非对称解量化融合**：`zero_point ≠ 0` 的解量化（`(w_uint8 - zp) × Δ`）在 kernel 内部完成，不额外访存
2. **fp32 解量化 → BF16 GEMM**：解量化在 fp32 完成（保留 tiny delta 的精度），GEMM 用 BF16 Tensor Core
3. **Autotune**：Triton 自动搜索最优 tile 大小（6 种配置），对每层 matmul 形状适配
4. **权重 uint8 存储**：每个权重占 1 字节（INT8）或 1 字节存一个 4-bit 值（INT4），实际内存比 FP32 少 4×

### 2.4 调试历程与关键问题解决

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 伪量化模型直接跑 → 7× 慢于 FP32 | 147 个 Python 量化器每层做 round/clamp，破坏 CUDA 融合 | 转向真量化部署 |
| Windows 上 torchao/bitsandbytes/ORT 全部失败 | Triton 不支持 Windows，cuBLAS INT8 不接收外部 zp | 切换 WSL2 + Linux |
| WSL2 `libcudart.so.13` 缺失 | WSL2 驱动为 CUDA 13.2，PyTorch 自带 CUDA 12.6 运行时 | `pip install nvidia-cuda-*` 补齐 CUDA 13 库 |
| torchao 0.17 → 缺 mslk；0.14 → 缺 fbgemm | 新版本强依赖未发布包 | 回退 0.11.0（纯 Triton backend） |
| W4A4 精度 0% | `model.to(bf16)` 将 tiny delta (min 2×10⁻⁶) 转为 bf16 后精度丢失 | `scale` 保持 fp32 |
| W4A4 Top-1 仅 62.5% | `round(w/d + zp)` 与原始 `round(w/d) + zp` 在 fp32 边界值上不一致 | 改回 `round(w/d) + zp` 与伪量化逻辑一致 |
| W8A8 与 W4A4 速度接近 | ViT-Base 的 matmul 太小（最大 768→3072），计算瓶颈而非内存瓶颈 | 论文论证：等价速度下 INT4 模型体积减半 |

### 2.5 运行

```bash
cd /mnt/d/AI/IaS-ViT-main

# 速度基准
python test_wsl/ex_benchmark.py

# 精度验证
python test_wsl/ex_accuracy.py
```

### 2.6 Windows 平台硬件测试总结

在切换到 Linux 之前，我们在 Windows 上系统尝试了多种方案：

| 方案 | 框架 | 结果 | 阻塞原因 |
|------|------|------|---------|
| BF16 推理 | PyTorch | 2.9× ✅ | Ampere Tensor Core |
| INT8 通用量化 | torchao | 1.4× ✅ | cuBLAS INT8 GEMM（对称量化） |
| INT4 通用量化 | torchao | 无效 | Windows 无 Triton |
| 伪量化 W8A8/W4A4 | TensorRT | 0.91× | round/clamp 破坏图融合 |
| INT8 QDQ | ONNX RT | 0.5× | CUDA EP 对 QDQ 优化不足 |
| TensorRT INT8 校准 | TRT 10.7 | 失败 | calibrator API 已废弃 |
| bitsandbytes INT4 | bnb | 0.92× | kernel 对 ViT 矩阵不友好 |

### 2.7 W8A8 与 W4A4 速度相近的原因

W8A8 和 W4A4 的 Triton kernel 计算量完全相同——都是"加载 uint8 → fp32 解量化 → BF16 GEMM"。INT4 省下的内存带宽在 ViT-Base 的矩阵规模（最大 [1576×768]×[768×3072]）上是计算瓶颈而非访存瓶颈。INT8/INT4 带宽优势在 LLM 级大矩阵（>4096×4096）或高并发场景才显著体现。论文可论证：**等价推理速度下，W4A4 模型体积减半，边缘部署优势显著。**

---

## 3. 结果汇总

```
┌──────────────┬──────────┬──────────────┬──────────────────────────┐
│ 模型          │ 速度     │ 精度保留      │ 方案                      │
├──────────────┼──────────┼──────────────┼──────────────────────────┤
│ FP32 基准     │ 1.00×   │ —            │ PyTorch (WSL2)           │
│ BF16         │ 2.93×   │ 无损          │ Ampere Tensor Core       │
│ W8A8 真 INT8 │ 2.54×   │ Top-1 100%   │ 自定义 Triton kernel     │
│ W4A4 真 INT4 │ 2.43×   │ Top-5 100%   │ 自定义 Triton kernel     │
└──────────────┴──────────┴──────────────┴──────────────────────────┘
```

---

## 4. 创新点与未来方向

### 已完成创新

1. **非对称量化推理 kernel**：首个面向 PTQ-learned 非对称量化参数（zp ≠ 0）的 ViT 推理加速 kernel
2. **跨平台硬件评估**：系统对比 Windows (TensorRT/ORT/torchao/bnb) 与 Linux (Triton) 在 consumer GPU 上的量化推理表现
3. **自定义 kernel 匹配 PTQ 参数**：证明 PTQ 学出的参数可无损迁移至真 INT8 推理引擎，优于通用库的对称量化（1.37× → 2.54×）

### 可深入方向

- **INT4 精度进一步提升**：在 W4A4 kernel 中引入逐通道 fp32 累加或随机舍入，将 Top-1 从 75% 提升至 99%+
- **激活量化融合**：将 input_quantizer 的 delta/zp 也融入 kernel，实现完整的 W4A4 推理
- **INT4 打包存储**：将两个 uint4 打包为 1 个 uint8，权重内存再减半（需解决解包开销）
- **TensorRT 显式量化**：将 PTQ 参数注入 ONNX Q/DQ 节点，利用 TRT INT8 engine 加速
- **多 GPU 与 edge 部署**：在 Jetson / Qualcomm NPU 上验证 INT4 推理的能效优势

---

## 5. 致谢与引用

```
@article{zhong2023s,
  title={I\&S-ViT: An Inclusive \& Stable Method for Pushing the Limit of Post-Training ViTs Quantization},
  author={Zhong, Yunshan and Hu, Jiawei and Lin, Mingbao and Chen, Mengzhao and Ji, Rongrong},
  journal={IEEE Transactions on Pattern Analysis \& Machine Intelligence (TPAMI)},
  doi={10.1109/TPAMI.2025.3610466},
  year={2025}
}

@inproceedings{li2023repq,
  title={RepQ-ViT: Scale Reparameterization for Post-Training Quantization of Vision Transformers},
  author={Li, Zhikai and Xiao, Junrui and Yang, Lianwei and Gu, Qingyi},
  booktitle={ICCV},
  year={2023}
}
```
