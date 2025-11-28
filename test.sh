#!/bin/bash

# 测试脚本 - 用于评估SAM3异常检测模型

# 注意：请确保已安装Python 3.12+和PyTorch 2.7.0+，并启用CUDA 12.6支持
# 推荐使用以下命令安装依赖：
# pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# pip install -r requirements.txt

# 设置默认参数
DATASET="mvtec"
TARGET_DATASET="visa"  # 零样本模式下的目标数据集
CATEGORY="bottle"
SPLIT="test"
MODEL_SIZE="base"
DEVICE="cuda"
THRESHOLD=0.5
OUTPUT_DIR="./output"
PRETRAINED_WEIGHTS="~/autodl-tmp/pre-training/sam3.pth"
CONFIG_FILE="./configs/base_config.yaml"
VISUALIZE=true
SAVE_VISUALIZATIONS=true
ZERO_SHOT=false
CHECKPOINT=""

# 解析命令行参数
if [ $# -ge 1 ]; then
    DATASET=$1
fi

if [ $# -ge 2 ]; then
    if [ "$2" = "zero-shot" ]; then
        ZERO_SHOT=true
        TARGET_DATASET=$1
        if [ $# -ge 3 ]; then
            CATEGORY=$3
        fi
    else
        CATEGORY=$2
    fi
fi

if [ $# -ge 3 ] && [ "$2" != "zero-shot" ]; then
    SPLIT=$3
fi

if [ $# -ge 4 ] && [ "$2" != "zero-shot" ]; then
    if [ "$4" = "zero-shot" ]; then
        ZERO_SHOT=true
        TARGET_DATASET=$DATASET
        DATASET="mvtec"  # 默认为MVTec作为源数据集
    fi
fi

if [ $# -ge 5 ] && [ "$2" = "zero-shot" ]; then
    THRESHOLD=$5
fi

# 确保输出目录存在
mkdir -p "${OUTPUT_DIR}/results"
mkdir -p "${OUTPUT_DIR}/visualizations"
mkdir -p "${OUTPUT_DIR}/logs"

# 打印测试配置
if [ "$ZERO_SHOT" = true ]; then
    cat << EOF
========================================
SAM3异常检测 - 零样本测试脚本
========================================
源数据集: ${DATASET}
目标数据集: ${TARGET_DATASET}
类别: ${CATEGORY}
模型大小: ${MODEL_SIZE}
设备: ${DEVICE}
阈值: ${THRESHOLD}
预训练权重: ${PRETRAINED_WEIGHTS}
配置文件: ${CONFIG_FILE}
输出目录: ${OUTPUT_DIR}
可视化: ${VISUALIZE}
保存可视化结果: ${SAVE_VISUALIZATIONS}
========================================
EOF
else
    cat << EOF
========================================
SAM3异常检测 - 测试脚本
========================================
数据集: ${DATASET}
类别: ${CATEGORY}
分割: ${SPLIT}
模型大小: ${MODEL_SIZE}
设备: ${DEVICE}
阈值: ${THRESHOLD}
预训练权重: ${PRETRAINED_WEIGHTS}
配置文件: ${CONFIG_FILE}
输出目录: ${OUTPUT_DIR}
可视化: ${VISUALIZE}
保存可视化结果: ${SAVE_VISUALIZATIONS}
========================================
EOF
fi

# 构建命令
CMD="python scripts/test.py \
    --config ${CONFIG_FILE} \
    --device ${DEVICE} \
    --model-size ${MODEL_SIZE} \
    --threshold ${THRESHOLD} \
    --output-dir ${OUTPUT_DIR}"

# 添加零样本模式参数
if [ "$ZERO_SHOT" = true ]; then
    CMD="${CMD} \
    --zero-shot \
    --source-dataset ${DATASET} \
    --target-dataset ${TARGET_DATASET} \
    --category ${CATEGORY}"
else
    CMD="${CMD} \
    --dataset ${DATASET} \
    --category ${CATEGORY} \
    --split ${SPLIT}"
fi

# 添加预训练权重或检查点
if [ ! -z "$CHECKPOINT" ]; then
    CMD="${CMD} --checkpoint ${CHECKPOINT}"
else
    CMD="${CMD} --pretrained-weights ${PRETRAINED_WEIGHTS}"
fi

# 添加可视化标志
if [ "$VISUALIZE" = true ]; then
    CMD="${CMD} --visualize"
fi

if [ "$SAVE_VISUALIZATIONS" = true ]; then
    CMD="${CMD} --save-visualizations"
fi

# 执行测试
echo "开始测试..."
echo "命令: ${CMD}"
echo "========================================"

# 记录开始时间
START_TIME=$(date +%s)

# 执行Python命令
$CMD

# 检查命令执行状态
if [ $? -ne 0 ]; then
    echo "测试失败！请检查错误信息。"
    exit 1
fi

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# 计算测试时间
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

# 打印测试完成信息
if [ "$ZERO_SHOT" = true ]; then
    cat << EOF
========================================
零样本测试完成！
测试时间: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒
结果保存至: ${OUTPUT_DIR}/results/
可视化结果保存至: ${OUTPUT_DIR}/visualizations/
EOF
else
    cat << EOF
========================================
测试完成！
测试时间: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒
结果保存至: ${OUTPUT_DIR}/results/
可视化结果保存至: ${OUTPUT_DIR}/visualizations/
EOF
fi

# 提示用户可用的后续操作
cat << EOF

可用的后续操作:

1. 查看结果: 打开 ${OUTPUT_DIR}/results/ 目录查看详细评估结果
2. 查看可视化: 打开 ${OUTPUT_DIR}/visualizations/ 目录查看可视化结果
3. 运行其他测试:
   - 常规测试: bash test.sh [数据集] [类别] [分割]
   - 零样本测试: bash test.sh [源数据集] zero-shot [类别]
   - 跨数据集零样本: bash test.sh [源数据集] [目标数据集] [类别] zero-shot

4. 运行单图像推理: python scripts/demo.py --image-path path/to/image.jpg
EOF
