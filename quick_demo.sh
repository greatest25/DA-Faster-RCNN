#!/bin/bash
# DA2OD 快速演示脚本 (使用少量数据)
# 用途: 快速验证完整流程

set -e

echo "=========================================="
echo "  DA2OD 快速演示流程"
echo "=========================================="
echo ""

# 步骤 1: 生成少量伪标签 (用于演示)
echo "步骤 1: 生成伪标签 (10张图像演示)..."
python generate_pseudo_labels.py \
    --weights output/resnet50_fpn_baseline/model_final.pth \
    --dataset city_trainT \
    --output pseudo_labels/demo_pseudo.json \
    --score-threshold 0.7 \
    --max-images 10

echo "✓ 伪标签已生成"
echo ""

# 步骤 2: 分析伪标签
echo "步骤 2: 分析伪标签质量..."
python scripts/analyze_pseudo_labels.py \
    --json pseudo_labels/demo_pseudo.json \
    --output-dir output_demo_analysis

echo "✓ 质量分析完成: output_demo_analysis/"
echo ""

# 步骤 3: 可视化
echo "步骤 3: 可视化伪标签..."
python scripts/visualize_pseudo_labels.py \
    --json pseudo_labels/demo_pseudo.json \
    --image-root datasets/cityscape/train_t \
    --output-dir output_demo_vis \
    --num-samples 5

echo "✓ 可视化完成: output_demo_vis/"
echo ""

# 步骤 4: 评估基线
echo "步骤 4: 评估基线模型..."
python evaluate_da2od.py \
    --weights output/resnet50_fpn_baseline/model_final.pth \
    --dataset city_testT \
    --output-dir output_demo_eval

echo "✓ 基线评估完成: output_demo_eval/eval_results.txt"
echo ""

echo "=========================================="
echo "  演示完成!"
echo "=========================================="
echo ""
echo "生成的文件:"
echo "  - pseudo_labels/demo_pseudo.json"
echo "  - output_demo_analysis/ (分析报告)"
echo "  - output_demo_vis/ (可视化图像)"
echo "  - output_demo_eval/eval_results.txt (评估结果)"
echo ""
echo "查看评估结果:"
echo "  cat output_demo_eval/eval_results.txt"
echo ""
