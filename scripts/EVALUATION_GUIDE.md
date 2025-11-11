# 模型评估指南

本文档说明如何评估训练好的模型，包括标准模型和 DA2OD 训练的模型。

## 📊 评估脚本对比

### 1. `evaluate.py` - 原始评估脚本
基础评估脚本，适用于标准 Faster R-CNN 模型。

**特点**:
- ✅ 简单直接
- ✅ 支持 COCO 和 VOC 格式
- ❌ 不支持 DA2OD EMA 权重
- ❌ 不支持伪标签数据集评估

**使用方法**:
```bash
python evaluate.py \
    --config-file configs/baseline_config.yaml \
    --weights output/model_final.pth \
    --dataset city_testT \
    --output-dir output_eval
```

---

### 2. `evaluate_da2od.py` - 增强评估脚本（推荐）
支持 DA2OD 和标准模型的增强评估脚本。

**特点**:
- ✅ 支持标准 Faster R-CNN 模型
- ✅ 支持 DA2OD EMA 权重加载
- ✅ 支持伪标签数据集评估
- ✅ 自动检测 DA2OD 可用性
- ✅ 更详细的结果输出

**使用方法**:
```bash
# 1. 评估标准模型
python evaluate_da2od.py \
    --weights output/model_final.pth \
    --dataset city_testT

# 2. 评估 DA2OD 模型（使用 EMA 权重）
python evaluate_da2od.py \
    --config-file configs/da2od_config.yaml \
    --weights output/model_final.pth \
    --dataset city_testT \
    --use-ema

# 3. 评估伪标签数据集
python evaluate_da2od.py \
    --weights output/model_final.pth \
    --dataset city_trainT \
    --eval-pseudo pseudo_labels/city_trainT_full_pseudo_thr07_coco.json
```

---

## 🎯 常见评估场景

### 场景 1: 评估基线模型（源域训练）
```bash
python evaluate_da2od.py \
    --weights output_baseline/model_final.pth \
    --dataset city_testT \
    --output-dir output_eval_baseline
```

### 场景 2: 评估使用伪标签训练的模型
```bash
python evaluate_da2od.py \
    --weights output_pseudo/model_final.pth \
    --dataset city_testT \
    --output-dir output_eval_pseudo
```

### 场景 3: 评估 DA2OD 训练的模型
```bash
# 方式 1: 使用 EMA 权重（推荐）
python evaluate_da2od.py \
    --config-file configs/da2od_config.yaml \
    --weights output_da2od/model_final.pth \
    --dataset city_testT \
    --use-ema \
    --output-dir output_eval_da2od_ema

# 方式 2: 使用 Student 权重
python evaluate_da2od.py \
    --config-file configs/da2od_config.yaml \
    --weights output_da2od/model_final.pth \
    --dataset city_testT \
    --output-dir output_eval_da2od_student
```

### 场景 4: 在不同数据集上评估
```bash
# 在源域测试
python evaluate_da2od.py \
    --weights output/model_final.pth \
    --dataset city_trainS

# 在目标域测试
python evaluate_da2od.py \
    --weights output/model_final.pth \
    --dataset city_testT

# 在伪标签数据上评估（检查伪标签质量）
python evaluate_da2od.py \
    --weights output/model_final.pth \
    --eval-pseudo pseudo_labels/city_trainT_full_pseudo_thr07_coco.json
```

---

## 📈 评估指标说明

### COCO 指标（推荐）
如果数据集使用 COCO 格式，会输出以下指标：

- **AP**: Average Precision @ IoU=0.50:0.95（主要指标）
- **AP50**: Average Precision @ IoU=0.50
- **AP75**: Average Precision @ IoU=0.75
- **APs**: AP for small objects (area < 32²)
- **APm**: AP for medium objects (32² ≤ area < 96²)
- **APl**: AP for large objects (area ≥ 96²)
- **AR@1**: Average Recall with 1 detection per image
- **AR@10**: Average Recall with 10 detections per image
- **AR@100**: Average Recall with 100 detections per image

### VOC 指标
如果数据集使用 VOC 格式，会输出：

- **mAP**: Mean Average Precision @ IoU=0.50
- 每个类别的 AP

---

## 🔍 结果解读

### 评估输出
运行评估后，会在终端显示并保存结果到 `output_eval/eval_results.txt`：

```
============================================================
Evaluation Results
============================================================

Dataset: city_testT
Weights: output/model_final.pth
EMA: False

bbox:
  AP: 45.234
  AP50: 68.567
  AP75: 48.123
  APs: 23.456
  APm: 47.890
  APl: 58.234
  ...
```

### 性能对比参考

典型 Cityscapes → Foggy Cityscapes 域适应性能：

| 方法 | AP50 |
|------|------|
| Source Only (无适应) | ~30-35% |
| Pseudo-labeling | ~40-45% |
| DA-Faster-RCNN | ~45-50% |
| **DA2OD** | **~55-58%** |

---

## 💡 最佳实践

### 1. 评估时机
- ✅ 训练完成后立即评估
- ✅ 每个重要 checkpoint 都评估
- ✅ 对比不同配置的模型

### 2. 评估数据集选择
- **city_testT**: 目标域测试集（主要评估指标）
- **city_trainS**: 源域验证（检查源域性能）
- **伪标签数据**: 验证伪标签质量

### 3. EMA vs Student 权重
对于 DA2OD 训练的模型：
- **EMA 权重**: 通常性能更好，更稳定（推荐用于最终评估）
- **Student 权重**: 训练中的主模型（用于监控训练进度）

建议**两者都评估**并对比结果。

### 4. 多次评估取平均
如果模型使用了随机性（如 dropout），建议：
```bash
# 运行 3-5 次评估，取平均值
for i in {1..3}; do
    python evaluate_da2od.py \
        --weights output/model_final.pth \
        --dataset city_testT \
        --output-dir output_eval_run_$i
done
```

---

## 🔧 故障排除

### 问题 1: CUDA Out of Memory
**解决**: 减少 batch size（在配置文件中调整 `TEST.IMS_PER_BATCH`）

### 问题 2: 权重加载失败
**错误**: `KeyError: 'ema'` 或类似

**解决**:
- 确认模型是否用 DA2OD 训练（如果不是，不要使用 `--use-ema`）
- 检查权重文件是否完整

### 问题 3: 数据集未注册
**错误**: `AssertionError: Dataset 'xxx' is not registered!`

**解决**:
- 确保运行了 `register_city_datasets()`
- 检查数据集名称是否正确
- 对于伪标签数据，使用 `--eval-pseudo` 参数

### 问题 4: EMA 不可用
**警告**: `⚠ DA2OD modules not available`

**原因**: DA2OD 模块未正确安装或导入失败

**解决**:
- 检查 `da2od/` 目录是否存在
- 确认 Python 路径包含项目根目录

---

## 📝 快速参考

### 常用命令

```bash
# 快速评估（使用默认设置）
python evaluate_da2od.py --weights output/model_final.pth

# DA2OD 完整评估
python evaluate_da2od.py \
    --config-file configs/da2od_config.yaml \
    --weights output/model_final.pth \
    --use-ema \
    --output-dir output_eval_da2od

# 对比评估（EMA vs Student）
python evaluate_da2od.py --weights output/model_final.pth --use-ema --output-dir eval_ema
python evaluate_da2od.py --weights output/model_final.pth --output-dir eval_student

# 伪标签质量检查
python evaluate_da2od.py \
    --weights output/model_final.pth \
    --eval-pseudo pseudo_labels/city_trainT_full_pseudo_thr07_coco.json
```

### 参数速查

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config-file` | 配置文件路径 | COCO R50-FPN |
| `--weights` | 模型权重路径 | 必需 |
| `--dataset` | 评估数据集名称 | city_testT |
| `--output-dir` | 输出目录 | ./output_eval |
| `--device` | 计算设备 | cuda |
| `--use-ema` | 使用 EMA 权重 | False |
| `--eval-pseudo` | 伪标签 JSON 路径 | None |

---

## 📚 相关文档

- 训练指南: `README.md`
- 伪标签生成: `scripts/README.md`
- DA2OD 配置: `configs/da2od_config.yaml`
- 数据集注册: `register_cityscapes.py`

---

**更新日期**: 2025-11-11
**维护者**: DA-Faster-RCNN Project
