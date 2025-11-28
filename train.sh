#!/bin/bash

# 训练脚本 - 用于微调SAM3模型进行异常检测

# 注意：请确保已安装Python 3.12+和PyTorch 2.7.0+，并启用CUDA 12.6支持
# 推荐使用以下命令安装依赖：
# pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# pip install -r requirements.txt

# 设置默认参数
DATASET="mvtec"
CATEGORY="bottle"
BATCH_SIZE=4
NUM_EPOCHS=50
LEARNING_RATE=5e-5
MODEL_SIZE="base"
DEVICE="cuda"
OUTPUT_DIR="./output"
PRETRAINED_WEIGHTS="~/autodl-tmp/pre-training/sam3.pth"
CONFIG_FILE="./configs/base_config.yaml"
VISUALIZE=false

# 解析命令行参数
if [ $# -ge 1 ]; then
    DATASET=$1
fi

if [ $# -ge 2 ]; then
    CATEGORY=$2
fi

if [ $# -ge 3 ]; then
    BATCH_SIZE=$3
fi

if [ $# -ge 4 ]; then
    NUM_EPOCHS=$4
fi

if [ $# -ge 5 ]; then
    LEARNING_RATE=$5
fi

if [ $# -ge 6 ]; then
    MODEL_SIZE=$6
fi

if [ $# -ge 7 ]; then
    DEVICE=$7
fi

# 确保输出目录存在
mkdir -p "${OUTPUT_DIR}/checkpoints"
mkdir -p "${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}/visualizations"

# 打印训练配置
cat << EOF
========================================
SAM3异常检测 - 训练脚本
========================================
数据集: ${DATASET}
类别: ${CATEGORY}
批量大小: ${BATCH_SIZE}
训练轮数: ${NUM_EPOCHS}
学习率: ${LEARNING_RATE}
模型大小: ${MODEL_SIZE}
设备: ${DEVICE}
预训练权重: ${PRETRAINED_WEIGHTS}
配置文件: ${CONFIG_FILE}
输出目录: ${OUTPUT_DIR}
可视化: ${VISUALIZE}
========================================
EOF

# 构建命令
CMD="python scripts/train.py \
    --config ${CONFIG_FILE} \
    --dataset ${DATASET} \
    --category ${CATEGORY} \
    --batch-size ${BATCH_SIZE} \
    --num-epochs ${NUM_EPOCHS} \
    --learning-rate ${LEARNING_RATE} \
    --model-size ${MODEL_SIZE} \
    --device ${DEVICE} \
    --pretrained-weights ${PRETRAINED_WEIGHTS} \
    --output-dir ${OUTPUT_DIR} \
    --fine-tune"

# 添加可视化标志
if [ "$VISUALIZE" = true ]; then
    CMD="${CMD} --visualize --save-visualizations"
fi

# 执行训练
echo "开始训练..."
echo "命令: ${CMD}"
echo "========================================"

# 记录开始时间
START_TIME=$(date +%s)

# 执行Python命令
$CMD

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# 计算训练时间
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

# 打印训练完成信息
cat << EOF
========================================
训练完成！
训练时间: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒
检查点保存至: ${OUTPUT_DIR}/checkpoints/
日志保存至: ${OUTPUT_DIR}/logs/
EOF

# 提示用户如何进行测试
cat << EOF

要测试训练后的模型，请运行:
  bash test.sh ${DATASET} ${CATEGORY} test --checkpoint ${OUTPUT_DIR}/checkpoints/best_model.pth
EOF
