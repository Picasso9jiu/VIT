# IaS-ViT: Post-Training Quantization for Vision Transformers

复现 TPAMI 2025 论文 *"I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization"*，包含算法复现、ONNX 模型导出、硬件加速测试。

---

## 项目结构

```
IaS-ViT-main/
├── quant/                     # 量化核心模块
│   ├── quantizer.py           #   UniformQuantizer / LogSqrt2Quantizer
│   ├── quant_modules.py       #   QuantConv2d / QuantLinear / QuantMatMul
│   ├── quant_model.py         #   模型包装与量化状态切换
│   ├── block_recon.py         #   Block 级输出重建
│   ├── layer_recon.py         #   Layer 级输出重建
│   └── data_utils.py          #   数据加载工具
├── utils/
│   ├── build_model.py         #   ViT 模型构建 (替换 Attention MatMul)
│   └── build_dataset.py       #   ImageNet 数据集加载与预处理
├── test_quant.py              # 【主脚本】PTQ 量化 pipeline 与精度验证
├── to_onnx.py                 #   ONNX 模型导出 (FP32 / W8A8 / W4A4)
├── model_download.py          #   下载预训练权重
├── test/                      # 硬件加速测试脚本
│   ├── test_speed.py          #   伪量化模型三档测速 (FP32 / W8A8 / W4A4)
│   ├── ex_hw_test.py          #   完整硬件基准 (FP32 / FP16 / BF16 / INT8 / TF32)
│   ├── ex_torchAO.py          #   torchao 真 INT4/INT8 量化测试
│   ├── ex_ort_quant.py        #   ONNX Runtime INT8/FP16 量化测试
│   └── ex_trt_test.py         #   TensorRT 部署测速
├── dataset/                   #  ImageNet 数据集 (不纳入版本控制)
├── *.pth / *.onnx             #  模型权重与 ONNX 文件 (不纳入版本控制)
└── .gitignore
```

---

## 1. 软件算法复现 ✅ (已完成)

### 1.1 环境依赖

- Python 3.11, PyTorch 2.8.0 (CUDA 12.6)
- timm, torchvision, onnx, onnxruntime-gpu, tensorrt 10.7.0
- bitsandbytes 0.48.2, torchao 0.11.0

```bash
pip install timm torchvision onnx onnxruntime-gpu tensorrt==10.7.0
pip install bitsandbytes torchao onnxconverter_common
```

### 1.2 下载模型权重

```bash
python model_download.py          # 下载 vit_base_full_pretrained.pth (~346MB)
```

### 1.3 运行 PTQ 量化与精度验证

```bash
python test_quant.py --model vit_base --dataset dataset/imagenet \
                     --w_bit 4 --a_bit 4 --w_cw --iter 1000
```

**IaS-ViT 三阶段量化流程：**
1. **Stage 1**: Q-Act + FP-Weight + Block/Layer 输出重建
2. **Stage 2**: 重参数化 — 将 qkv/fc1 的 per-channel 量化参数合并为标量
3. **Stage 3**: Q-Act + Q-Weight + 重新优化

### 1.4 导出 ONNX 模型

```bash
python to_onnx.py --fp32                                    # FP32 全精度
python to_onnx.py --model vit_base_w8a8.pth --w_bits 8 --w_cw  # W8A8 量化
python to_onnx.py --model vit_base_w4a4.pth --w_bits 4 --w_cw  # W4A4 量化
```

**注意：** 当前导出的 W8A8/W4A4 ONNX 为**伪量化**模型——权重仍以 FP32 存储，仅在前向传播中插入 round/clamp 算子。这保持了量化精度，但不产生真实的推理加速。

---

## 2. 硬件加速测试 🔄 (进行中)

### 2.1 测试环境

| 项目 | 规格 |
|------|------|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU (4 GB) |
| 架构 | Ampere (SM 8.6) |
| OS | Windows 11 Pro |
| CUDA | 12.6 |

### 2.2 已尝试方案及结果

| 方案 | 框架 | 精度 | 结果 | 原因 |
|------|------|------|------|------|
| BF16 推理 | PyTorch | BF16 | **2.9x 加速** ✅ | Ampere Tensor Core |
| FP16 推理 | PyTorch / TensorRT | FP16 | **2.9x 加速** ✅ | Tensor Core |
| TF32 推理 | PyTorch | TF32 | **1.7x 加速** ✅ | Tensor Core (FP32 matmul) |
| INT8 权重量化 | torchao | INT8 | **1.4x 加速** ✅ | cuBLAS INT8 GEMM |
| INT4 权重量化 | torchao | INT4 | **无效** (≈FP32) | Windows 无 Triton fused kernel |
| 伪量化 W8A8 | ONNX RT / TensorRT | FP32 | 0.47x (ONNX RT), 0.91x (TRT) | round/clamp op 破坏图融合 |
| INT8 QDQ | ONNX Runtime | INT8 | 0.5x (CUDA EP) | Q/DQ 节点优化不足 |
| INT8 校准 | TensorRT | INT8 | **失败** | TRT 10.x calibrator API 废弃 |
| INT4 权重量化 | bitsandbytes | INT4 | **倒退** (0.92x) | Kernel 对 ViT 不友好 |

### 2.3 关键发现

1. **Consumer GPU (RTX 3050) 上 BF16/FP16 是最佳部署格式**——利用 Ampere Tensor Core 获得 3x 加速，零代码改动
2. **INT8 有真实加速但有限**——torchao INT8 通过 cuBLAS INT8 GEMM 获得 1.4x 加速
3. **伪量化模型不能直接提速**——ONNX 中的 round/clamp op 破坏了推理引擎的 kernel 融合，必须转换为真实 INT8/FP16 格式
4. **INT4 在 Windows 上不可行**——缺少 Triton / CUDA C++ 扩展 / TensorRT INT4 支持
5. **TensorRT INT8 校准在 TRT 10.x Python API 中已废弃**——需改用显式量化 (Q/DQ in ONNX) 或降级到旧版本

### 2.4 下一步计划

- [ ] **Linux 环境 (WSL2)** — 安装 Ubuntu + PyTorch + Triton，跑 torchao INT4 kernel
- [ ] **TensorRT INT8** — 在 Linux 上使用 `trtexec` CLI 或旧版 TRT Python API 完成 INT8 校准
- [ ] **端到端对比** — FP32 / W8A8 / W4A4 的精度-速度-模型大小三维评估
- [ ] **TensorRT 显式量化** — 将 IaS-ViT 量化参数注入 ONNX Q/DQ 节点，导出兼容 TRT 的 INT8 模型

---

## 3. 引用

```
@article{zhong2023s,
  title={I\&S-ViT: An Inclusive \& Stable Method for Pushing the Limit of Post-Training ViTs Quantization},
  author={Zhong, Yunshan and Hu, Jiawei and Lin, Mingbao and Chen, Mengzhao and Ji, Rongrong},
  journal={IEEE Transactions on Pattern Analysis \& Machine Intelligence (TPAMI)},
  doi={10.1109/TPAMI.2025.3610466},
  year={2025}
}
```

## 致谢

```
@inproceedings{li2023repq,
  title={RepQ-ViT: Scale Reparameterization for Post-Training Quantization of Vision Transformers},
  author={Li, Zhikai and Xiao, Junrui and Yang, Lianwei and Gu, Qingyi},
  booktitle={ICCV},
  year={2023}
}
```
