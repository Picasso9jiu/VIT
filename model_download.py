import torch
import timm

# 🔥 直接下载和你项目完全匹配的 ViT-Base 全精度模型
# 模型名：vit_base_patch16_224 就是你量化用的 vit_base
model = timm.create_model('vit_base_patch16_224', pretrained=True)

# 保存到项目根目录，名字一目了然
torch.save(model.state_dict(), "vit_base_full_pretrained.pth")

print("="*50)
print("✅ 全精度模型下载 + 保存成功！")
print("📦 模型路径：D:/AI/IaS-ViT-main/vit_base_full_pretrained.pth")
print("📏 模型大小：约 340MB (全精度原始模型)")
print("="*50)