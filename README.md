# 零样本工业瑕疵检测项目

本项目基于Meta的SAM3（Segment Anything Model 3）实现了零样本工业瑕疵检测系统，能够在无需额外训练的情况下，通过自然语言提示词检测工业产品中的各类瑕疵。

## 功能特性

- **零样本检测**：无需为特定数据集或瑕疵类型进行训练
- **多数据集支持**：兼容MVTec、VisA等常用工业瑕疵检测数据集
- **灵活提示词**：支持通过自然语言提示词引导模型检测特定类型的瑕疵
- **高性能评估**：提供完整的评估指标（IoU、精确率、召回率、F1分数等）
- **可视化结果**：自动生成检测结果可视化，便于分析和展示

## 评估指标

支持的评估指标包括：

### 基础指标
- **IoU (Intersection over Union)**：交并比，衡量预测掩码与真实掩码的重叠程度
- **精确率 (Precision)**：预测为缺陷的像素中，实际为缺陷的比例
- **召回率 (Recall)**：实际为缺陷的像素中，被正确预测的比例
- **F1分数**：精确率和召回率的调和平均值

### 高级指标
- **图像级AUROC**：评估模型区分正常图像和异常图像的能力
- **像素级AUROC**：评估模型在像素级别区分正常区域和异常区域的能力
- **AP (Average Precision)**：平均精度，综合考虑不同阈值下的精确率和召回率
- **AUPRO (Area Under PR curve by Region)**：按区域组织的PR曲线下面积，特别适合评估缺陷检测性能

## 可视化功能

系统提供多种可视化方式，帮助用户直观理解和分析检测结果：

### 基础可视化
- **缺陷叠加图**：在原始图像上以半透明红色叠加显示预测的缺陷区域
- **对比叠加图**：在原始图像上同时显示预测缺陷（红色）和真实缺陷（绿色）

### 高级可视化
- **热力图显示**：使用颜色梯度直观展示缺陷置信度分布
- **可视化网格**：创建包含原始图像、预测掩码、真实掩码等多视图对比的综合网格图
- **样本对比集**：自动生成前N个样本的对比可视化，便于批量分析

### 批量可视化汇总
- **可视化目录**：自动创建结构化的可视化结果目录
- **指标可视化**：可选的评估指标图表生成功能

可视化结果默认保存在`output/{时间戳}/visualizations/`目录下，包括基本可视化和高级网格可视化。

## 环境要求

- Python 3.12+
- PyTorch 2.7.0+
- CUDA 12.6+（推荐使用GPU加速）

## 安装步骤

1. 克隆项目仓库

```bash
git clone https://github.com/yourusername/SamAD3.git
cd SamAD3
```

2. 安装依赖包

```bash
pip install -r requirements.txt
```

3. 安装SAM3（需按照官方文档申请访问权限）

```bash
pip install sam3
```

## 配置说明

项目配置文件位于 `config.py`，可以根据需要修改以下参数：

- `DATASET_ROOT`：数据集根目录路径（默认：`~/autodl-tmp/datasets`）
- `MODEL_WEIGHTS_PATH`：预训练权重路径（默认：`~/autodl-tmp/pre-training`）
- `MODEL_CONFIG`：模型配置（设备、置信度阈值、IoU阈值等）

## 使用方法

### 1. 在MVTec数据集上运行检测

```bash
python scripts/run_mvtec_detection.py --category bottle
```

可选参数：
- `--category`：指定MVTec数据集的类别（如bottle、cable、capsule等）
- `--prompt`：自定义提示词，用于引导模型检测特定类型的瑕疵

### 2. 在VisA数据集上运行检测

```bash
python scripts/run_visa_detection.py --category candle
```

可选参数：
- `--category`：指定VisA数据集的类别（如candle、capsules、cashew等）
- `--prompt`：自定义提示词

### 3. 对单张图像进行检测

```bash
python scripts/detect_single_image.py --image_path path/to/your/image.jpg
```

必需参数：
- `--image_path`：输入图像的路径

可选参数：
- `--prompt`：自定义提示词
- `--no_visualization`：不保存可视化结果

## 项目结构

```
SamAD3/
├── models/                 # 模型相关代码
│   └── sam3_model.py       # SAM3模型封装
├── data_loaders/           # 数据加载相关代码
│   ├── dataset_loader.py   # 数据集加载器
│   └── data_pipeline.py    # 数据预处理管道
├── utils/                  # 工具函数
│   ├── metrics.py          # 评估指标计算
│   ├── image_processing.py # 图像处理函数
│   └── data_augmentation.py # 数据增强函数
├── scripts/                # 示例脚本
│   ├── run_mvtec_detection.py  # MVTec数据集检测
│   ├── run_visa_detection.py   # VisA数据集检测
│   └── detect_single_image.py  # 单张图像检测
├── config.py               # 配置文件
├── requirements.txt        # 依赖列表
├── defect_detector.py      # 主程序
└── README.md               # 项目说明
```

## 提示词指南

有效的提示词可以显著提高检测性能。以下是一些提示词示例：

- **通用瑕疵**："defect flaw scratch damage imperfection irregularity"
- **玻璃/塑料瓶**："broken crack scratch damage contamination defect chip"
- **布料/纺织品**："stain discoloration damage defect pattern irregularity hole"
- **金属表面**："scratch cut damage defect dent corrosion imperfection"

可以根据具体的检测场景和目标瑕疵类型调整提示词。

## 结果说明

检测结果保存在 `outputs/results_时间戳/` 目录下，包括：

- **可视化图像**：原始图像、预测掩码和真实掩码（如果有）的叠加显示
- **metrics.txt**：详细的评估指标和统计信息

## 注意事项

1. 使用前请确保已正确安装SAM3并获得预训练权重的访问权限
2. 数据集路径需按照配置文件中的设置正确组织
3. 在不同的硬件上，推理速度可能会有所差异
4. 对于复杂的瑕疵检测场景，可以尝试调整提示词以获得更好的效果

## 许可证

本项目采用MIT许可证。

## 致谢

- Meta Research for SAM3
- 所有使用的开源库和数据集的作者和维护者
