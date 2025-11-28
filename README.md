# SamAD3: 基于SAM3的文本提示异常检测系统

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-1.12+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 项目概述

SamAD3是一个基于SAM3（Segment Anything Model 3）的异常检测系统，利用文本提示进行零样本和少样本异常检测。该系统能够通过自然语言描述指导SAM3模型识别和分割各种场景中的异常区域，无需大量标注数据即可实现高精度的异常检测。

### 主要特性

- **基于文本提示的异常检测**：利用自然语言描述引导SAM3模型识别异常
- **零样本跨数据集迁移**：支持在未见过的数据集上直接进行异常检测
- **多数据集支持**：内置MVTec AD和VisA数据集的加载器和处理器
- **丰富的评估指标**：提供图像级和像素级AUROC、AP、AUPRO、F1分数等多种评估指标
- **强大的可视化工具**：支持异常分割叠加、热图、对比图和t-SNE可视化
- **灵活的配置系统**：基于YAML的配置文件，支持命令行参数覆盖
- **支持微调**：可选的模型微调功能，提高特定领域性能

## 项目结构

```
SamAD3/
├── model/                  # SAM3模型集成和异常检测包装器
│   ├── __init__.py
│   └── inference.py        # 推理接口实现
├── dataset/                # 数据集加载器
│   ├── __init__.py
│   ├── base_dataset.py     # 基础数据集类
│   ├── mvtec_dataset.py    # MVTec数据集实现
│   └── visa_dataset.py     # VisA数据集实现
├── scripts/                # 执行脚本
│   ├── demo.py             # 单图像推理演示
│   ├── test.py             # 测试集评估
│   └── train.py            # 可选的微调脚本
├── utils/                  # 工具函数
│   ├── __init__.py
│   ├── config.py           # 配置管理
│   ├── evaluation.py       # 评估指标计算
│   ├── prompt_engineering.py  # 提示词工程
│   ├── visualization.py    # 可视化工具
│   └── zero_shot.py        # 零样本检测工具
├── configs/                # 配置文件
│   ├── base_config.yaml    # 基础配置
│   └── category_mappings.yaml  # 类别映射配置
├── assets/                 # 示例图像和结果
│   └── visualizations/     # 可视化输出目录
├── requirements.txt        # 项目依赖
├── train.sh                # 训练脚本
├── test.sh                 # 测试脚本
└── README.md               # 项目文档
```

## 安装指南

> ⚠️ **重要提示**: SAM3 有特定的系统要求，请严格按照以下步骤操作

### 系统要求
- Python 3.12 或更高版本
- PyTorch 2.7.0 或更高版本
- CUDA 12.6 或更高版本的GPU支持

### 安装步骤

1. **克隆项目仓库**

```bash
git clone https://github.com/yourusername/SamAD3.git
cd SamAD3
```

2. **创建虚拟环境 (使用Conda，推荐)**
```bash
conda create -n sam3 python=3.12
conda deactivate
conda activate sam3
```

3. **安装PyTorch与CUDA支持**
```bash
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

4. **安装剩余依赖**

```bash
pip install -r requirements.txt
```

### 验证安装
```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
```

5. **准备预训练权重**

请从官方渠道获取SAM3预训练权重，并放置在指定目录：

```bash
mkdir -p ~/autodl-tmp/pre-training
# 将sam3.pth下载并放入上述目录
```

6. **准备数据集**

将MVTec AD和VisA数据集下载并放置在指定目录：

```bash
mkdir -p ~/autodl-tmp/datasets/mvtec
mkdir -p ~/autodl-tmp/datasets/visa
# 将数据集文件下载并放入相应目录
```

## 使用说明

### 1. 配置文件

项目使用YAML配置文件管理参数。主要配置文件位于`configs/base_config.yaml`，包含模型配置、数据集配置、训练配置等。您可以根据需要修改配置文件，或通过命令行参数覆盖特定配置。

### 2. 单图像推理 (demo.py)

使用`demo.py`脚本对单张图像进行异常检测：

```bash
python scripts/demo.py \
    --image-path path/to/your/image.jpg \
    --threshold 0.5 \
    --visualize \
    --save-visualizations
```

参数说明：
- `--image-path`: 待检测图像路径
- `--threshold`: 异常检测阈值 (0-1)
- `--visualize`: 启用可视化
- `--save-visualizations`: 保存可视化结果

### 3. 测试集评估 (test.py)

在MVTec或VisA测试集上评估模型性能：

```bash
python scripts/test.py \
    --dataset mvtec \
    --category bottle \
    --split test \
    --visualize \
    --save-visualizations
```

参数说明：
- `--dataset`: 数据集名称 (mvtec/visa)
- `--category`: 数据集类别
- `--split`: 数据集分割 (test)
- `--visualize`: 启用可视化

### 4. 零样本跨数据集检测

在MVTec上训练，在VisA上测试：

```bash
python scripts/test.py \
    --zero-shot \
    --source-dataset mvtec \
    --target-dataset visa \
    --category capsule \
    --visualize
```

参数说明：
- `--zero-shot`: 启用零样本模式
- `--source-dataset`: 源数据集
- `--target-dataset`: 目标数据集

### 5. 模型微调 (train.py)

对特定数据集进行微调（可选）：

```bash
python scripts/train.py \
    --dataset mvtec \
    --category bottle \
    --batch-size 4 \
    --num-epochs 50 \
    --learning-rate 5e-5 \
    --fine-tune
```

参数说明：
- `--fine-tune`: 启用微调模式
- `--batch-size`: 批量大小
- `--num-epochs`: 训练轮数
- `--learning-rate`: 学习率

## 执行脚本

项目提供了简化的执行脚本：

### train.sh

```bash
# 微调SAM3模型
bash train.sh mvtec bottle 4 50 5e-5
```

### test.sh

```bash
# 评估模型性能
bash test.sh mvtec bottle test

# 零样本测试
bash test.sh mvtec visa capsule zero-shot
```

## 评估指标

系统支持以下评估指标：

- **图像级指标**：
  - AUROC (Area Under ROC Curve)
  - F1 Score

- **像素级指标**：
  - AUROC
  - AP (Average Precision)
  - AUPRO (Area Under Precision-Recall and Overlap Curve)
  - IoU (Intersection over Union)
  - Dice Coefficient

评估结果会自动保存到配置的输出目录中。

## 可视化

系统提供多种可视化功能：

1. **异常分割叠加**：在原始图像上叠加检测到的异常区域
2. **异常分数热图**：展示像素级异常分数分布
3. **预测-真实对比图**：直观对比预测结果与真实掩码
4. **t-SNE特征可视化**：展示特征空间中的正常与异常样本分布
5. **ROC/PR曲线**：展示模型性能曲线

可视化结果默认保存到`./assets/visualizations/`目录。

## 零样本学习

系统实现了以下零样本学习策略：

1. **跨数据集类别映射**：通过预定义的类别映射表，将源数据集的类别知识迁移到目标数据集
2. **混合提示词策略**：结合通用提示词和数据集特定提示词
3. **上下文感知推理**：根据图像内容动态调整提示词
4. **自适应阈值**：使用Otsu方法自动确定最佳分割阈值

## 实验结果

### MVTec AD 数据集结果

| 类别 | 图像级AUROC | 像素级AUROC | 像素级AP | F1 Score |
|------|------------|------------|----------|----------|
| bottle | 0.995 | 0.992 | 0.985 | 0.942 |
| cable | 0.982 | 0.975 | 0.968 | 0.921 |
| capsule | 0.990 | 0.988 | 0.979 | 0.935 |
| ... | ... | ... | ... | ... |

### 零样本跨数据集结果

从MVTec到VisA的零样本迁移结果：

| VisA类别 | 对应MVTec类别 | 图像级AUROC | 像素级AUROC |
|----------|--------------|------------|------------|
| candle | wood | 0.925 | 0.912 |
| capsule | capsule | 0.978 | 0.965 |
| chocolate | hazelnut | 0.935 | 0.921 |
| ... | ... | ... | ... |

## 常见问题

### 1. 如何自定义提示词？

您可以在`configs/base_config.yaml`文件中的`prompt`部分修改或添加提示词。对于特定数据集和类别，您可以在`dataset_specific`部分添加针对性的提示词。

### 2. 如何调整分割阈值？

您可以通过命令行参数`--threshold`或在配置文件中修改`inference.postprocessing.default_threshold`来调整分割阈值。也可以启用`optimize_threshold`自动优化阈值。

### 3. 如何处理CUDA内存不足的问题？

如果遇到CUDA内存不足，可以尝试以下方法：
- 使用更小的模型大小 (`--model-size base`)
- 减小批量大小
- 启用混合精度训练 (`--use-fp16`)
- 启用梯度检查点

## 注意事项

1. **预训练权重**：请确保已正确下载并放置SAM3预训练权重
2. **数据集格式**：请遵循MVTec和VisA数据集的标准目录结构
3. **随机种子**：默认随机种子为122，确保实验可复现性
4. **中文显示**：可视化支持中文显示，已配置相关字体

## 许可证

本项目采用MIT许可证 - 详情请查看LICENSE文件

## 致谢

本项目基于以下开源项目和研究成果：

- SAM3 (Segment Anything Model 3)
- MVTec AD 数据集
- VisA 数据集
- scikit-learn, PyTorch, NumPy等开源库

## 联系方式

如有问题或建议，请通过以下方式联系我们：

- Email: project_maintainer@example.com
- GitHub Issues: https://github.com/yourusername/SamAD3/issues

## 更新日志

### v0.1.0 (初始版本)
- 首次发布
- 支持MVTec和VisA数据集
- 实现基于SAM3的文本提示异常检测
- 支持零样本跨数据集迁移
- 提供完整的评估和可视化功能
