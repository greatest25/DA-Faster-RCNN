#!/bin/bash
# DA2OD 完整流程演示脚本
# 作者: DA-Faster-RCNN Project
# 日期: 2025-11-11

set -e  # 遇到错误立即退出

PROJECT_ROOT="/mnt/lyh/DA-FasterCNN/DA-Faster-RCNN"
cd "${PROJECT_ROOT}"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# ============================================
# 步骤 0: 检查环境
# ============================================
print_step "步骤 0: 环境检查"

if [ ! -f "output/model_final.pth" ]; then
    echo "错误: 基线模型不存在 (output/model_final.pth)"
    echo "请先训练基线模型"
    exit 1
fi

print_success "基线模型存在: output/model_final.pth"

if [ ! -d "datasets/cityscape/train_t" ]; then
    echo "错误: 目标域数据集不存在"
    exit 1
fi

print_success "数据集检查通过"

# ============================================
# 步骤 1: 生成伪标签
# ============================================
print_step "步骤 1: 生成伪标签"

PSEUDO_JSON="pseudo_labels/city_trainT_full_pseudo_thr07_coco.json"

if [ -f "${PSEUDO_JSON}" ]; then
    print_warning "伪标签已存在,跳过生成步骤"
else
    print_success "开始生成伪标签..."
    python generate_pseudo_labels.py \
        --weights output/model_final.pth \
        --dataset city_trainT \
        --output "${PSEUDO_JSON}" \
        --score-threshold 0.7 \
        --max-images 0
    
    print_success "伪标签生成完成: ${PSEUDO_JSON}"
fi

# ============================================
# 步骤 2: 分析伪标签质量
# ============================================
print_step "步骤 2: 分析伪标签质量"

python scripts/analyze_pseudo_labels.py \
    --json "${PSEUDO_JSON}" \
    --output-dir output_analysis_pseudo

print_success "质量分析完成,查看: output_analysis_pseudo/"

# ============================================
# 步骤 3: 可视化伪标签样本
# ============================================
print_step "步骤 3: 可视化伪标签样本"

python scripts/visualize_pseudo_labels.py \
    --json "${PSEUDO_JSON}" \
    --image-root datasets/cityscape/train_t \
    --output-dir output_vis_pseudo \
    --num-samples 10

print_success "可视化完成,查看: output_vis_pseudo/"

# ============================================
# 步骤 4: 评估基线模型 (可选)
# ============================================
print_step "步骤 4: 评估基线模型 (可选)"

python evaluate_da2od.py \
    --weights output/model_final.pth \
    --dataset city_testT \
    --output-dir output_eval_baseline

print_success "基线评估完成,查看: output_eval_baseline/eval_results.txt"

# ============================================
# 步骤 5: 使用 DA2OD 训练 (演示版 - 少量迭代)
# ============================================
print_step "步骤 5: DA2OD 训练 (演示版)"

print_warning "注意: 完整训练需要较长时间"
print_warning "此演示仅运行 100 迭代用于测试流程"

# 备份原配置
cp configs/da2od_config.yaml configs/da2od_config.yaml.bak

# 修改为少量迭代用于演示
cat > configs/da2od_demo.yaml << 'DEMO_CFG'
# DA2OD 演示配置 (快速测试)
MODEL:
  BACKBONE:
    NAME: "build_resnet_fpn_backbone"
  ROI_HEADS:
    NUM_CLASSES: 8

SOLVER:
  MAX_ITER: 100  # 演示用,实际训练建议 5000+
  CHECKPOINT_PERIOD: 50
  IMS_PER_BATCH: 4
  BASE_LR: 0.0005

OUTPUT_DIR: "./output_da2od_demo"

# DA2OD 特定配置
DAOD:
  ENABLE_ALIGNER: True

EMA:
  ENABLED: True
  ALPHA: 0.999

PSEUDO_LABEL:
  SCORE_THRESH: 0.7
DEMO_CFG

print_success "演示配置已创建"
print_warning "如需完整训练,请修改 SOLVER.MAX_ITER 为 5000+"

# 注意: 实际训练命令 (注释掉,避免占用太多时间)
# python uda_train.py --config-file configs/da2od_demo.yaml

print_warning "训练命令(未执行):"
echo "  python uda_train.py --config-file configs/da2od_demo.yaml"

# ============================================
# 步骤 6: 评估 DA2OD 模型 (假设已训练)
# ============================================
print_step "步骤 6: 评估 DA2OD 模型 (假设已训练)"

print_warning "此步骤需要先完成训练"
print_warning "训练完成后运行:"
echo ""
echo "  # 评估 Student 模型"
echo "  python evaluate_da2od.py \\"
echo "      --config-file configs/da2od_config.yaml \\"
echo "      --weights output_da2od/model_final.pth \\"
echo "      --dataset city_testT \\"
echo "      --output-dir output_eval_da2od_student"
echo ""
echo "  # 评估 EMA 模型 (推荐)"
echo "  python evaluate_da2od.py \\"
echo "      --config-file configs/da2od_config.yaml \\"
echo "      --weights output_da2od/model_final.pth \\"
echo "      --dataset city_testT \\"
echo "      --use-ema \\"
echo "      --output-dir output_eval_da2od_ema"

# ============================================
# 总结
# ============================================
print_step "流程总结"

echo "已完成步骤:"
echo "  ✓ 步骤 0: 环境检查"
echo "  ✓ 步骤 1: 伪标签生成"
echo "  ✓ 步骤 2: 质量分析"
echo "  ✓ 步骤 3: 可视化"
echo "  ✓ 步骤 4: 基线评估"
echo ""
echo "待执行步骤:"
echo "  → 步骤 5: DA2OD 训练 (运行时间较长)"
echo "  → 步骤 6: DA2OD 评估"
echo ""
echo "生成的文件:"
echo "  - ${PSEUDO_JSON}"
echo "  - output_analysis_pseudo/"
echo "  - output_vis_pseudo/"
echo "  - output_eval_baseline/eval_results.txt"
echo ""
print_success "演示流程完成!"
