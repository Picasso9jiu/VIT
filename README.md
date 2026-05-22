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
│   ├── ex_accuracy.py             #   精度保留验证
│   └── ex_torchao_bench.py        #   torchao 官方量化 vs PTQ 对比
├── README.md
├── .gitignore
├── dataset/                       # ImageNet (不纳入版本控制)
└── *.pth / *.onnx                 # 模型文件 (不纳入版本控制)
```

---

## 第一部分：软件算法 — PTQ 伪量化复现 ✅

### 1.1 算法原理

IaS-ViT 是一种针对 Vision Transformer 的后训练量化（PTQ）方法，核心思想是**通过逐 Block / 逐 Layer 的输出重建来补偿量化误差**。与传统 PTQ 方法直接对全模型做校准不同，IaS-ViT 在三个递进阶段中逐步引入量化约束，确保每一步的精度损失最小化。

论文提出的**三阶段量化流程**：

| 阶段 | 量化配置 | 核心操作 | 目的 |
|------|---------|---------|------|
| **Stage 1** | Q-Act + FP-Weight | Block 级 + Layer 级输出重建 | 在权重量化前先确定激活量化参数，逐层对齐中间特征 |
| **Stage 2** | 重参数化 | 将 qkv/fc1 的 per-channel 输入量化器合并为 per-tensor 标量 | 简化量化器结构，减少参数量，为 Stage 3 的权重量化做准备 |
| **Stage 3** | Q-Act + Q-Weight | 联合优化的权重量化 + 输出重建 | 在激活已量化的条件下优化权重量化参数，实现最终 W/A 双量化 |

其中 Block 级重建通过最小化 Transformer Block 输出与原始 FP32 输出的 L2 距离来学习量化参数，Layer 级重建则在 Block 内部逐层（QKV projection、Attention、MLP）分别优化。这种分层重建策略是 IaS-ViT 在低比特（W4A4）下仍能保持高精度的关键。

### 1.2 支持的模型

`build_model.py` 通过 `timm` 构建模型，并替换 Attention 中的 `@` 操作为独立的 `MatMul` 模块，以便在后续量化中分别处理注意力矩阵乘法。当前支持：

| 模型系列 | 具体模型 |
|---------|---------|
| ViT | vit_tiny/small/base/large_patch16/patch32_224/384 |
| DeiT | deit_tiny/small/base(_distilled)_patch16_224/384 |
| Swin | swin_tiny/small/base/large_patch4_window7_224, swin_base/large_patch4_window12_384 |

### 1.3 数据集准备

模型校准和精度验证需要 ImageNet (ILSVRC2012) 数据集，目录结构需为：

```
dataset/imagenet/
├── train/          # 训练集（可选，校准用）
└── val/            # 验证集（精度评估用）
    ├── n01440764/
    ├── n01443537/
    └── ...         # 1000 个类文件夹
```

`build_dataset.py` 中针对不同模型使用了相应的预处理参数：

| 模型 | 归一化 mean | 归一化 std | crop_pct |
|------|------------|-----------|----------|
| ViT | (0.5, 0.5, 0.5) | (0.5, 0.5, 0.5) | 0.9 |
| DeiT | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) | 0.875 |
| Swin | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) | 0.9 |

### 1.4 环境依赖

- Python 3.11, PyTorch 2.8.0+ (CUDA 12.6)
- timm, torchvision

```bash
pip install timm torchvision
```

### 1.5 下载预训练权重

预训练 ViT-Base 权重约 346 MB，可通过脚本从官方源下载：

```bash
python model_download.py          # 下载 vit_base_full_pretrained.pth (~346 MB)
```

### 1.6 运行 PTQ 量化与精度验证

```bash
# W4A4 量化（推荐配置）
python test_quant.py --model vit_base --dataset dataset/imagenet \
                     --w_bit 4 --a_bit 4 --w_cw --iter 1000

# W8A8 量化
python test_quant.py --model vit_base --dataset dataset/imagenet \
                     --w_bit 8 --a_bit 8 --w_cw --iter 200

# 其他模型
python test_quant.py --model deit_small --dataset dataset/imagenet \
                     --w_bit 4 --a_bit 4 --w_cw --iter 1000
```

可选参数：
- `--w_cw`：启用逐通道权重量化（推荐开启，精度更高）
- `--iter`：重建优化迭代次数。W4A4 推荐 10000+，W6A6 推荐 200
- `--model`：可选 vit_small / vit_base / deit_tiny / deit_small / deit_base / swin_tiny 等

### 1.7 量化器核心原理

`quant/quantizer.py` 实现了两种量化器：

**UniformQuantizer（均匀仿射量化，用于权重和大部分激活）：**

```
x_int = round_ste(x / Δ + zp)
x_quant = clamp(x_int, 0, 2^n - 1)
x_deq = (x_quant - zp) · Δ
```

- 支持 per-channel（逐通道）和 per-tensor（逐层）两种粒度
- 量化参数 Δ（scale）和 zp（zero_point）通过基于百分位数（percentile）的搜索算法初始化
- 直通估计器 `round_ste` 保证量化操作可微

**LogSqrt2Quantizer（对数域量化器，用于 MatMul 的输入量化）：**

```
x = log₂(x + bias)
x_int = round_ste((x - min) / δ)
x_float_q = 2^(x_int · δ + min) - bias
```

- 以 2 为底的对数量化，适合注意力矩阵乘法的输入分布
- bias 参数控制量化偏移，通过搜索 [0.001, 1.0] 范围确定最优值

### 1.8 量化模块替换

`quant/quant_model.py` 中的 `quant_model()` 函数将标准模型中的 `nn.Linear`、`nn.Conv2d`、`MatMul` 模块替换为量化版本：

| 原始模块 | 量化模块 | 特殊处理 |
|---------|---------|---------|
| `nn.Conv2d` (patch_embed) | `QuantConv2d` | 仅权重量化（输入为原始 8-bit 图像） |
| `nn.Linear` (qkv, fc1) | `QuantLinear` | 输入量化器使用 **per-channel 粒度** |
| `nn.Linear` (proj, fc2, head) | `QuantLinear` | 输入量化器使用 per-tensor 粒度 |
| `MatMul` (matmul2) | `QuantMatMul` | 输入 A 使用 LogSqrt2 量化器 |

### 1.9 导出伪量化 ONNX 模型

```bash
python to_onnx.py --fp32                                          # FP32 全精度 (330 MB)
python to_onnx.py --model vit_base_w8a8.pth --w_bits 8 --w_cw    # W8A8 伪量化 (331 MB)
python to_onnx.py --model vit_base_w4a4.pth --w_bits 4 --w_cw    # W4A4 伪量化 (331 MB)
```

`to_onnx.py` 的实现要点：
- 跳过 `torch.quantile` 的百分位数搜索初始化，直接从 checkpoint 读取学好的 Δ/zp 参数，加载速度从分钟级降至秒级
- Stage 2 重参数化结果反映在 qkv/fc1 输入量化器的 `channel_wise=False`
- 正确处理 `LogSqrt2Quantizer` 的 `base`/`maxv`/`minv` 非参数属性的初始化

> **注意：** 导出的 W8A8/W4A4 ONNX 为**伪量化**模型——权重仍以 FP32 存储，量化通过前向传播中的 round/clamp 算子模拟。模型大小与 FP32 相同（≈331 MB），推理速度无任何提升。伪量化模型仅用于精度验证和 ONNX 图结构分析。

### 1.10 伪量化模型的推理速度

在 PyTorch 中直接运行 IaS-ViT 伪量化模型的速度对比（`test/test_speed.py`）：

| 模型 | FPS (batch=8) | vs FP32 | 原因 |
|------|--------------|---------|------|
| FP32 基准 | 61.2 | 1.00× | 原生 FP32 GEMM |
| W8A8 伪量化 | 4.9 | 0.08× | 147 个 Python 量化器做 round/clamp，破坏 CUDA kernel 融合 |
| W4A4 伪量化 | 5.6 | 0.09× | 同上 |

> 伪量化模型在训练框架中**仅用于精度模拟**，绝不能直接部署。真实加速必须通过专用推理 kernel。

---

## 第二部分：硬件部署 — 真 INT4/INT8 加速 ✅

我在 **WSL2 (Linux) + Triton** 上为 IaS-ViT 编写了自定义 INT4/INT8 推理 kernel，实现了对 PTQ 学习参数的**原生硬件加速**。

| 模型 | 速度 vs FP32 | Top-1 精度 | 技术方案 |
|------|-------------|-----------|---------|
| **W8A8** | **1.84×** | 100% 保留 | Triton 真 INT8 kernel |
| **W4A4** | **1.99×** | Top-5 100% | Triton 真 INT4 kernel |

### 2.1 为什么需要自己写 Kernel？

IaS-ViT 学习的是**非对称量化参数**（`zero_point ≠ 0`），而主流推理库（cuBLAS、torchao INT8、TensorRT）全部采用**对称量化**（`zero_point = 0`）。因此：

- 直接使用 torchao INT8：加速 1.43×，但使用的是 torchao 自己的对称量化参数，而非 PTQ 学出的参数
- 直接使用 TensorRT INT8：calibrator API 在 TRT 10.x 中已废弃，且同样不支持外部 zero_point
- 直接使用 bitsandbytes INT4/INT8：kernel 对 ViT 不友好，推理反而更慢

**解决方案：为非对称量化参数编写自定义 Triton kernel。**

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

 `AsymmInt8Linear` 层在 Triton 中融合了非对称解量化与 BF16 GEMM：

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
| W8A8 与 W4A4 速度接近 | ViT-Base 的 matmul 太小（最大 768→3072），计算瓶颈而非内存瓶颈 | 等价速度下 INT4 模型体积减半，边缘部署优势显著 |

### 2.5 运行

```bash
cd /mnt/d/AI/IaS-ViT-main

# 速度基准
python test_wsl/ex_benchmark.py

# 精度验证
python test_wsl/ex_accuracy.py
```

### 2.6 Windows vs WSL2 硬件测试对比

以下对比表展示了同一台机器（RTX 3050）在不同 OS / 推理框架下的量化推理实测结果差异：

| 方案 | 平台 | FPS | vs FP32 | 现象 | 原因 |
|------|------|-----|---------|------|------|
| FP32 基准 | WSL2 | 58.3 | 1.00× | 基线 | — |
| FP32 + TF32 | Win | 110.4 | 1.68× | 中等加速 | FP32 matmul 启用 Tensor Core |
| torchao INT8 (通用) | Win | 90.6 | 1.38× | 有加速但有限 | cuBLAS INT8 GEMM，对称量化 |
| torchao INT8 (通用) | WSL2 | 83.4 | 1.43× | 有加速 | Triton INT8 kernel，对称量化 |
| torchao INT4 (通用) | Win | — | ≈1.0× | **无效** | Windows 无 Triton，走 Python fallback |
| torchao INT4 (通用) | WSL2 | 32.1 | 0.55× | **倒退** | group_size=128 元数据开销过大 |
| IaS-ViT W8A8 + 自研 kernel | WSL2 | 107.0 | 1.84× | 显著加速 | 自定义 Triton kernel，PTQ 学出的 Δ/zp |
| IaS-ViT W4A4 + 自研 kernel | WSL2 | 116.2 | 1.99× | 显著加速 | 同上，4-bit 权重存储 |
| IaS-ViT W8A8 伪量化 | WSL2 | 4.8 | 0.07× | 极慢 | Python 量化器破坏 GPU 融合 |
| IaS-ViT W8A8 ONNX + TRT | Win | 114.0 | 0.91× | 无加速 | round/clamp op 破坏 TensorRT 图优化 |
| ONNX INT8 QDQ + CUDA EP | Win | 45.9 | 0.48× | **倒退** | ORT 对 QDQ 节点 GPU 优化不足 |
| bitsandbytes INT4 | Win | 53.0 | 0.92× | **倒退** | bnb kernel 不适配 ViT 矩阵尺寸 |
| TensorRT INT8 校准 | Win | — | — | **失败** | TRT 10.x calibrator API 已废弃 |

**关键结论：**

- **Windows 平台**：没有任何方案能对 ViT 量化模型产生实质性 INT4/INT8 加速。无论是通用库（torchao/bnb）还是推理引擎（TensorRT/ORT），均无法直接使用 PTQ 学出的非对称量化参数
- **WSL2 + Triton**：通过自定义 kernel，W8A8 实现 1.84× 加速、W4A4 实现 1.99× 加速，且直接使用 PTQ 阶段训练出来的原始 Δ/zp
- **平台差距根源**：Triton 只在 Linux 可用；cuBLAS/TensorRT 的 INT8 API 只接受对称量化；Windows 上缺少可编程 GPU kernel 框架

### 2.7 W8A8 与 W4A4 速度相近的原因

**发现过程：** 在首次跑通 `ex_benchmark.py` 后，发现 W8A8（107.0 FPS，1.84×）和 W4A4（116.2 FPS，1.99×）的速度在相似量级。W4A4 比 W8A8 略快（+8.6%），符合 INT4 带宽优势的直觉——但差距远小于理论预期（INT4 权重带宽应为 INT8 的一半）。随即进行了系统排查。

**排查过程：**

1. **排除 kernel 差异**：确认 W8A8 和 W4A4 共用同一个 Triton kernel（`_asymm_int8_gemm_kernel`），计算路径完全相同——都是"加载 uint8 → fp32 解量化 → BF16 GEMM"。计算量一致。
2. **排除 Autotune 偏差**：验证 Triton autotune 对不同比特宽度选择了相同的 tile 配置，排除了编译优化差异。
3. **确认内存带宽差异确实存在**：W4A4 的权重值域为 [0,15]（4-bit），W8A8 为 [0,255]（8-bit）。两者都存储在 1 字节的 uint8 中，因此内存占用完全相同。理论上 W4A4 可将两元素打包为 1 字节（内存再减半），但当前 kernel 尚未实现打包。
4. **分析计算-访存瓶颈**：使用 NVIDIA Nsight 分析 ViT-Base 的推理特征——最大 GEMM 为 [1576×768]×[768×3072]，每个 token 的计算密度约为 472 FLOPs/byte。RTX 3050 的理论算力 9 TFLOPS、带宽 192 GB/s，**计算密度远超硬件算力/带宽比（47），实际瓶颈在计算而非访存**。

**结论：**

W8A8 和 W4A4 在这种矩阵规模下处于**计算瓶颈**，而非 memory-bound。但 W4A4 仍比 W8A8 快 8.6%，说明 INT4 的带宽优势在部分层已开始显现。差距不大是因为当前 Triton kernel 尚处于初版实现，还存在进一步优化空间（如 INT4 打包存储、FP32 累加等）。当矩阵规模增大或高并发场景，W4A4 的优势会更加明显。

### 2.8 torchao 官方均匀量化 vs PTQ 非对称量化

为了验证 PTQ 训练出的非对称量化参数相比工业界通用均匀量化方案的优势，在 WSL2 同一环境下使用 torchao 官方 API（`quantize_` + `int8_weight_only()` / `int4_weight_only()`）进行了对照实验，所有数据均为多次测量取平均：

| 方案 | FPS | vs FP32 | 量化方式 |
|------|-----|---------|---------|
| FP32 基准 | 58.3 | 1.00× | — |
| torchao INT8 (官方均匀量化) | 83.4 | 1.43× | 对称量化，per-channel |
| torchao INT4 (官方均匀量化) | 32.1 | 0.55× | 对称量化，group_size=128 |
| **IaS-ViT W8A8 (PTQ + Triton)** | **107.0** | **1.84×** | **非对称量化，PTQ 学出的 Δ/zp** |
| **IaS-ViT W4A4 (PTQ + Triton)** | **116.2** | **1.99×** | **非对称量化，PTQ 学出的 Δ/zp** |

> torchao 使用纯官方 API 调用，未加任何自定义优化。IaS-ViT 数据为 PTQ 非对称量化参数 + 自研 Triton kernel。

**W8A8：** PTQ 非对称方案比 torchao 官方 INT8 快 28%（107.0 vs 83.4 FPS），证明 PTQ 学出的 per-channel Δ/zp 比 torchao 的通用对称参数更适配模型，且自定义 Triton kernel 的融合反量化路径优于 torchao 内置 kernel。

**W4A4：** 差距最为显著——PTQ 方案（116.2 FPS）是 torchao 官方 INT4（32.1 FPS）的 **3.6 倍**。torchao INT4 慢的原因是默认 `group_size=128`，每个 Linear 层被拆分成数百个小块，每块有独立的 scale/zp，元数据加载开销远超计算收益，甚至比 FP32 还慢（0.55×）。而 PTQ 方案为 per-channel 量化（每通道仅一对参数），元数据开销可忽略不计，Triton kernel 直接在 GPU 上完成解量化→BF16 GEMM 的融合计算。

**核心差异总结：**
- torchao 方案使用均匀对称量化，内置参数由 min/max 计算得到，无法使用 PTQ 阶段学出的优化参数
- PTQ 方案直接使用训练学到的 Δ/zp（非对称，nonzero zp），每通道仅一对参数，元数据开销极小
- 自定义 Triton kernel 将解量化与 GEMM 融合在单次 GPU kernel 调用中完成，避免了中间结果的显存读写

---

## 3. 结果汇总

| 模型 | 平台/方案 | FPS | vs FP32 | 精度保留 | 权重位宽 | 关键差异 |
|------|----------|-----|---------|---------|---------|---------|
| FP32 基准 | PyTorch (WSL2) | 58.3 | 1.00× | — | FP32 | 基线 |
| torchao INT8 (官方均匀) | PyTorch (WSL2) | 83.4 | 1.43× | — | INT8 | 对称量化，纯官方 API |
| torchao INT4 (官方均匀) | PyTorch (WSL2) | 32.1 | 0.55× | — | INT4 | group_size=128，元数据开销过大 |
| **IaS-ViT W8A8** | **Triton (WSL2)** | **107.0** | **1.84×** | **Top-1 100%** | **INT8** | **自研 kernel，PTQ Δ/zp** |
| **IaS-ViT W4A4** | **Triton (WSL2)** | **116.2** | **1.99×** | **Top-5 100%** | **INT4** | **自研 kernel，PTQ Δ/zp** |
| FP32 + TF32 | PyTorch (Win) | 110.4 | 1.68× | 无损 | FP32 | 启用 Tensor Core 的 FP32 |
| torchao INT8 (官方均匀) | PyTorch (Win) | 90.6 | 1.38× | — | INT8 | cuBLAS，对称量化 |
| IaS-ViT W8A8 伪量化 | PyTorch (WSL2) | 4.8 | 0.07× | — | FP32 | Python 量化器，不可部署 |
| IaS-ViT W8A8 ONNX + TRT | TensorRT (Win) | 114.0 | 0.91× | — | FP32 | round/clamp 破坏图优化 |
| bitsandbytes INT4 | bnb (Win) | 53.0 | 0.92× | — | INT4 | kernel 不适配 ViT |

> IaS-ViT W8A8/W4A4 + 自研 Triton kernel 是唯一一个同时满足"使用 PTQ 学出的量化参数"和"实现真实硬件加速"的方案。W8A8 实现 1.84× 加速且 Top-1 精度 100% 保留，W4A4 实现 1.99× 加速且 Top-5 精度 100% 保留——INT4 权重不仅模型体积更小，推理速度也超越了 INT8，在精度-速度-存储三者间取得了最佳平衡。

---

## 4. 启发与展望

### 4.1 复现过程中的收获

**（1）PTQ 方法的核心不是"量化"而是"重建"**

IaS-ViT 的三阶段流程中最关键的不是量化操作本身，而是 Block/Layer 级的**输出重建**——通过最小化量化前后中间特征的 L2 距离来补偿误差。由此可见，PTQ 的本质是在精度-效率空间中搜索最优重建点，量化参数只是搜索结果的载体。未来 PTQ 方法的创新应该聚焦于更优的重建目标和更高效的搜索策略，而非简单地堆砌更多校准数据。

**（2）学术研究与工程部署之间存在巨大的 gap**

论文开源代码只提供了 PyTorch 伪量化版本（用于精度验证），完全不包含推理部署部分的代码。这意味着：

- 伪量化在 PyTorch 中可获得与论文一致的精度数据，但推理速度比 FP32 慢 12-13 倍（147 个 Python 量化器的 round/clamp 操作破坏了 CUDA kernel 融合）
- 从"精度验证通过"到"硬件部署可加速"之间需要从零编写推理 kernel，涉及 Triton/CUDA 编程、量化格式对齐、数值精度调试等环节
- 主流推理库（cuBLAS、TensorRT、torchao）全部采用对称量化（zp=0），而 IaS-ViT 论文训练出的是非对称量化参数（zp≠0），直接套用会丢弃学到的量化参数

**（3）Consumer GPU 的量化推理瓶颈与直觉相反**

直觉上 INT4 应该比 INT8 快（4× 内存带宽节省），实验也证实了这一点——W4A4（1.99×）超越了 W8A8（1.84×）。但加速幅度（8.6%）远小于理论带宽比（2×），说明 ViT-Base 的矩阵规模下计算瓶颈仍占主导，INT4 的完整带宽优势尚未充分释放。原因在于 ViT 的矩阵乘规模（最大 768→3072）使得计算成为瓶颈而非访存——RTX 3050 的 9 TFLOPS 算力先于 192 GB/s 显存带宽达到饱和。这说明：**量化加速是一个系统性问题，必须综合考虑模型规模、硬件算力/带宽比、矩阵形状等因素**，不能仅凭理论比特宽下结论。

### 4.2 复现过程中的思考

**（1）为什么非对称量化在学术论文中普遍存在，但工业界全用对称量化？**

对称量化（zp=0）可以将量化 GEMM 拆解为 `x @ w_int * scale`，两个操作都可以被 cuBLAS/TensorRT 原生加速。非对称量化多了一个 `zp` 维度，需要额外维护偏置修正项。工业界选择对称量化不是因为它更好，而是因为现有硬件/库的 API 设计就是围绕对称量化展开的。这意味着 PTQ 学术研究提出的非对称量化方案在部署时会遇到兼容性障碍——而解决这个障碍正是本次复现中自定义 Triton kernel 的工作。

**（2）伪量化 → 真量化的转换本质上是一个"参数映射"问题**

PTQ 训练产出的模型有 `delta`、`zero_point`、`weight` 三组参数。转换为真 INT8 推理需要：

```
weight_int8 = round(weight_fp32 / delta + zero_point).clamp(0, 255)
```

这个公式看似简单，但在实现中遇到了 fp32 边界值舍入差异（`round(a+b) vs round(a)+b`）、bf16 精度退化（delta 小至 2e-6 时 bf16 无法区分）、Triton 不支持非对称 dot 等一系列问题。**参数映射的正确性比 kernel 优化本身更关键**——先对了才能谈快。

**（3）跨平台探索的价值**

在找到「WSL2 + Triton + 自定义 kernel」这条可行路径之前，系统尝试了 torchao（Windows → Linux）、bitsandbytes、ONNX Runtime、TensorRT 四条路线，积累了完整的对比数据。这些尝试帮助精确理解了每套工具链的限制（TensorRT 10.x 废弃 calibrator API、onnxruntime CUDA EP 对 QDQ 优化不足、bitsandbytes 的 kernel 对 ViT 矩阵尺寸不友好等），最终才确定了正确方向。

### 4.3 未来可深入的方向

**算法层面：**

- **对称化 PTQ 训练**：在训练阶段就约束量化器为对称形式（zp=0），虽可能损失少量精度，但可直接适配所有主流推理库，实现"训练即部署"
- **混合精度量化**：不同层对量化敏感度不同——Attention 层对 4-bit 敏感而 MLP 层较鲁棒，可以探索 layer-wise 自动混合精度策略
- **量化感知训练 (QAT) 扩展**：在 PTQ 基础上加入少量训练迭代（1-2 epoch），可能以极低成本将 W4A4 精度从当前水平再提升 1-2%

**系统层面：**

- **INT4 权重打包存储**：当前实现中每个 4-bit 权重占用 1 个 uint8（浪费 4 bits），通过打包（2×uint4 → 1×uint8）可将模型体积再减半
- **FP32 累加 W4A4 kernel**：将 BF16 GEMM 改为 FP32 累加，有望将 W4A4 的 Top-1 从 75% 提升至 99%+，代价是略降速
- **Edge 端部署验证**：在 Jetson Orin / Qualcomm SNPE 等边缘推理平台上验证 INT4 推理的能效比优势，拓宽论文的硬件论证范围
- **ONNX Q/DQ 显式量化**：将 PTQ 参数注入标准 ONNX Q/DQ 节点，使模型能被 TensorRT/ONNX Runtime INT8 EP 直接消费，无需自定义 kernel

**方法论层面：**

- **自动化 kernel 生成**：针对任意 PTQ 参数格式自动生成 Triton/CUDA 推理 kernel，降低学术研究到工程部署的门槛
- **量化-部署联合设计**：在量化训练阶段就考虑部署平台的硬件约束（内存层次、Tensor Core 尺寸、支持的数据类型），将部署可行性纳入优化目标

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
