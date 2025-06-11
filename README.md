

# Vision Transformer (ViT) —— 基础 PyTorch 实现（MNIST 数据集）

## 📌 简介

本项目提供了一个轻量级、简明的 Vision Transformer (ViT) 基础实现，专为初学者和研究人员设计，便于理解 ViT 模型的基本原理及其在小型图像分类任务（如 MNIST）上的应用。

**参考实现**：[lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)  
本仓库实现了简化版本，方便理解 ViT 结构，不涉及复杂优化或 COCO 等大型数据集，如需扩展至 COCO 格式数据，可自行修改 `dataset.py` 及相关处理逻辑，或者咨询 GPT 进一步扩展。

---

## 🚀 项目特点

- 自定义 Vision Transformer 实现
- 自动下载 MNIST 数据（若未检测到 `datasets/MNIST/`）
- 包含 **训练 / 验证 / 测试** 完整流程
- 支持保存 **上一轮(last)权重** 和 **最佳(best)权重**
- 采用命令行参数（CLI）控制
- 支持 CPU 环境（默认配置）

---

## 🖥️ 环境要求

| 软件包      | 版本                  |
|-------------|----------------------|
| Python      | ≥ 3.8                |
| PyTorch     | 2.0.0                |
| torchvision | 0.15.1               |
| einops      | 0.6.0                |
| tqdm        | 4.65.0               |

安装依赖（推荐在虚拟环境或 Conda 环境中）：
```bash
pip install torch torchvision einops tqdm
```

---

## ⚙️ 训练参数配置（`config.py`）

| 参数名            | 值         | 说明                          |
|-------------------|------------|-------------------------------|
| image_size        | 28         | MNIST 图像尺寸                |
| patch_size        | 7          | Patch 切分大小                |
| num_classes       | 10         | 分类类别数（数字 0-9）        |
| dim              | 64         | Patch Embedding 维度          |
| depth            | 6          | Transformer Block 层数        |
| heads            | 8          | Multi-head Attention 头数     |
| mlp_dim          | 128        | MLP 层隐藏单元数              |
| dim_head         | 32         | 每个 Head 的维度              |
| dropout          | 0.1        | Attention Dropout            |
| emb_dropout      | 0.1        | Embedding Dropout            |
| batch_size       | 64         | 每批样本数                    |
| epochs           | 100        | 训练轮次     |
| lr               | 3e-4       | 学习率                        |
| device           | cpu        | 训练设备                      |

---

## 📂 目录结构

```
vit_mnist/
├── vit_pytorch/
│   └── vit.py             # ViT 模型实现
├── dataset.py             # MNIST 数据集加载
├── train.py               # 训练 / 验证 / 测试逻辑（含 checkpoint）
├── utils.py               # 工具函数（准确率、保存加载模型）
├── config.py              # 配置文件（超参数）
├── main.py                # CLI 入口
└── checkpoints/           # 自动保存权重（last, best）
```

---

## 📌 运行示例（命令行）

### 1. 训练（100轮）

```bash
# 🚀如果检测到 datasets文件夹为空，会自动下载
python main.py --mode train
```

- 训练过程自动保存：
  - `checkpoints/last_checkpoint.pth` —— 每轮更新
  - `checkpoints/best_checkpoint.pth` —— 验证准确率最佳

---

### 2. 验证

```bash
python main.py --mode validate
```

输出示例：

```
✅ Loaded checkpoint from checkpoints/best_checkpoint.pth, epoch 7, acc 0.9830
[Val] Accuracy: 0.9902
```

---

### 3. 测试

```bash
python main.py --mode test
```

输出示例：

```
✅ Loaded checkpoint from checkpoints/best_checkpoint.pth, epoch 7, acc 0.9830
[Test] Accuracy: 0.9805
```

---

## 📌 注意事项

- 当前版本仅适用于 **MNIST 数据**，如需适配 COCO 格式，请修改 `dataset.py` 以及 `DataLoader` 部分（GPT 可协助）。
- 本项目支持 CPU 训练及推理，**无需 GPU** 即可完整复现结果。
- 本项目为 ViT 学习及基础实验设计，未做复杂调参、数据增强或正则化，适合教学与初步实验。


---

## 📌 参考项目

- https://github.com/lucidrains/vit-pytorch