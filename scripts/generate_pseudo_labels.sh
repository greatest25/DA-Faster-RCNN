#!/bin/bash
# 伪标签生成便捷脚本
# 用途: 从训练好的模型生成伪标签用于域适应训练

set -e  # 遇到错误立即退出

# ==================== 配置区域 ====================
PROJECT_ROOT="/mnt/lyh/DA-FasterCNN/DA-Faster-RCNN"
MODEL_PATH="${PROJECT_ROOT}/output/model_final.pth"
CONFIG_PATH="${PROJECT_ROOT}/configs/da2od_config.yaml"
OUTPUT_DIR="${PROJECT_ROOT}/pseudo_labels"
DATASET_NAME="city_trainT"

# 伪标签生成参数
SCORE_THRESHOLD=0.7
MAX_IMAGES=0  # 0 表示处理所有图像
OUTPUT_FILENAME="city_trainT_full_pseudo_thr07_coco.json"

# ==================== 函数定义 ====================
print_header() {
    echo ""
    echo "=========================================="
    echo "  $1"
    echo "=========================================="
}

print_info() {
    echo "✓ $1"
}

print_error() {
    echo "✗ ERROR: $1" >&2
}

check_file() {
    if [ ! -f "$1" ]; then
        print_error "文件不存在: $1"
        return 1
    fi
    return 0
}

# ==================== 主流程 ====================
print_header "伪标签生成流程"

# 1. 检查环境
print_info "检查环境..."
cd "${PROJECT_ROOT}" || exit 1

if ! check_file "${MODEL_PATH}"; then
    echo "提示: 你可以指定其他模型路径，例如:"
    echo "  export MODEL_PATH=/path/to/your/model.pth"
    exit 1
fi

if ! check_file "${CONFIG_PATH}"; then
    echo "警告: 配置文件不存在，将使用默认配置"
    CONFIG_PATH=""
fi

# 2. 创建输出目录
mkdir -p "${OUTPUT_DIR}"
print_info "输出目录: ${OUTPUT_DIR}"

# 3. 生成伪标签
print_header "生成伪标签"
OUTPUT_PATH="${OUTPUT_DIR}/${OUTPUT_FILENAME}"

echo "参数配置:"
echo "  - 模型: ${MODEL_PATH}"
echo "  - 配置: ${CONFIG_PATH:-使用默认}"
echo "  - 数据集: ${DATASET_NAME}"
echo "  - 置信度阈值: ${SCORE_THRESHOLD}"
echo "  - 图像数量: ${MAX_IMAGES} (0=全部)"
echo "  - 输出文件: ${OUTPUT_PATH}"
echo ""

if [ -n "${CONFIG_PATH}" ]; then
    python generate_pseudo_labels.py \
        --config-file "${CONFIG_PATH}" \
        --weights "${MODEL_PATH}" \
        --dataset-name "${DATASET_NAME}" \
        --output "${OUTPUT_PATH}" \
        --score-threshold ${SCORE_THRESHOLD} \
        --max-images ${MAX_IMAGES}
else
    python generate_pseudo_labels.py \
        --weights "${MODEL_PATH}" \
        --dataset-name "${DATASET_NAME}" \
        --output "${OUTPUT_PATH}" \
        --score-threshold ${SCORE_THRESHOLD} \
        --max-images ${MAX_IMAGES}
fi

# 4. 验证生成结果
if [ ! -f "${OUTPUT_PATH}" ]; then
    print_error "伪标签生成失败: ${OUTPUT_PATH}"
    exit 1
fi

print_header "验证伪标签"
FILE_SIZE=$(du -h "${OUTPUT_PATH}" | cut -f1)
print_info "文件大小: ${FILE_SIZE}"

# 使用 Python 快速统计
python -c "
import json
with open('${OUTPUT_PATH}', 'r') as f:
    data = json.load(f)
print(f'✓ 图像数量: {len(data.get(\"images\", []))}')
print(f'✓ 标注数量: {len(data.get(\"annotations\", []))}')
print(f'✓ 类别数量: {len(data.get(\"categories\", []))}')
if data.get('annotations'):
    avg_score = sum(ann.get('score', 0) for ann in data['annotations']) / len(data['annotations'])
    print(f'✓ 平均置信度: {avg_score:.3f}')
"

# 5. 完成提示
print_header "生成完成"
echo "伪标签已生成: ${OUTPUT_PATH}"
echo ""
echo "后续步骤:"
echo "  1. 分析质量: python scripts/analyze_pseudo_labels.py --json ${OUTPUT_PATH}"
echo "  2. 可视化: python scripts/visualize_pseudo_labels.py --json ${OUTPUT_PATH}"
echo "  3. 训练: python uda_train.py --config-file configs/da2od_config.yaml"
echo ""
