# 伪标签生成和分析工具

这个目录包含三个用于伪标签生成、可视化和分析的便捷脚本。

## 📦 脚本清单

### 1. `generate_pseudo_labels.sh` - 伪标签生成脚本
一键生成伪标签的 Bash 脚本，自动处理完整流程。

**功能**:
- 自动检查模型和配置文件
- 调用 `generate_pseudo_labels.py` 生成伪标签
- 验证生成结果并显示统计信息
- 提供后续步骤指引

**使用方法**:
```bash
# 基本用法（使用默认配置）
bash scripts/generate_pseudo_labels.sh

# 自定义模型路径
export MODEL_PATH=/path/to/your/model.pth
bash scripts/generate_pseudo_labels.sh
```

**配置参数**（可在脚本中修改）:
- `MODEL_PATH`: 模型权重路径（默认: `output/model_final.pth`）
- `CONFIG_PATH`: 配置文件路径（默认: `configs/da2od_config.yaml`）
- `SCORE_THRESHOLD`: 置信度阈值（默认: 0.7）
- `MAX_IMAGES`: 处理图像数量（0 = 全部）
- `OUTPUT_FILENAME`: 输出文件名

---

### 2. `visualize_pseudo_labels.py` - 伪标签可视化脚本
可视化伪标签并保存带标注的图像。

**功能**:
- 随机选择样本图像
- 在图像上绘制边界框和类别标签
- 显示置信度分数
- 保存可视化结果

**使用方法**:
```bash
# 基本用法
python scripts/visualize_pseudo_labels.py

# 自定义参数
python scripts/visualize_pseudo_labels.py \
    --json pseudo_labels/city_trainT_full_pseudo_thr07_coco.json \
    --image-root datasets/cityscape/train_t \
    --output-dir output_vis_pseudo \
    --num-samples 20 \
    --random-seed 42
```

**参数说明**:
- `--json`: 伪标签 JSON 文件路径
- `--image-root`: 图像根目录
- `--output-dir`: 可视化结果保存目录
- `--num-samples`: 可视化图像数量（默认: 10）
- `--random-seed`: 随机种子（默认: 42）

**输出**:
- 保存在 `output_vis_pseudo/` 目录
- 文件命名格式: `vis_001_<image_name>.png`

---

### 3. `analyze_pseudo_labels.py` - 伪标签质量分析脚本
深入分析伪标签质量，生成详细统计报告和可视化图表。

**功能**:
- 置信度分布分析（均值、中位数、标准差等）
- 类别分布统计
- 每张图像标注数量分析
- 边界框尺寸分布（按 COCO 标准分类）
- 生成统计图表和文本报告

**使用方法**:
```bash
# 基本用法
python scripts/analyze_pseudo_labels.py

# 自定义参数
python scripts/analyze_pseudo_labels.py \
    --json pseudo_labels/city_trainT_full_pseudo_thr07_coco.json \
    --output-dir output_analysis_pseudo
```

**参数说明**:
- `--json`: 伪标签 JSON 文件路径
- `--output-dir`: 分析结果保存目录

**输出文件**:
- `confidence_distribution.png`: 置信度分布直方图
- `category_distribution.png`: 类别分布条形图
- `analysis_report.txt`: 详细文本报告

**分析内容**:
1. **置信度分布**:
   - 统计量：平均值、中位数、标准差、最小/最大值
   - 区间分布：[0.5-0.6), [0.6-0.7), ..., [0.9-1.0)
   
2. **类别分布**:
   - 每个类别的标注数量和占比
   - 降序排列
   
3. **每张图像标注统计**:
   - 有/无标注的图像数量
   - 平均标注数、中位数、最小/最大值
   
4. **边界框尺寸**:
   - 平均/中位数面积
   - COCO 尺寸分类（Small/Medium/Large）

---

## 🚀 完整工作流程

### 步骤 1: 生成伪标签
```bash
bash scripts/generate_pseudo_labels.sh
```

### 步骤 2: 分析质量
```bash
python scripts/analyze_pseudo_labels.py \
    --json pseudo_labels/city_trainT_full_pseudo_thr07_coco.json
```

### 步骤 3: 可视化检查
```bash
python scripts/visualize_pseudo_labels.py \
    --json pseudo_labels/city_trainT_full_pseudo_thr07_coco.json \
    --num-samples 20
```

### 步骤 4: 训练使用
生成的伪标签会自动在 `uda_train.py` 中被检测和使用（需要取消注释伪标签加载代码）。

```bash
python uda_train.py --config-file configs/da2od_config.yaml
```

---

## 📊 输出目录结构

```
DA-Faster-RCNN/
├── pseudo_labels/
│   └── city_trainT_full_pseudo_thr07_coco.json  # 生成的伪标签
├── output_analysis_pseudo/
│   ├── confidence_distribution.png              # 置信度分布图
│   ├── category_distribution.png                # 类别分布图
│   └── analysis_report.txt                      # 文本报告
└── output_vis_pseudo/
    ├── vis_001_<image>.png                      # 可视化图像
    ├── vis_002_<image>.png
    └── ...
```

---

## 💡 使用建议

### 调整置信度阈值
根据分析结果调整生成伪标签时的置信度阈值：

```bash
# 在 generate_pseudo_labels.sh 中修改
SCORE_THRESHOLD=0.8  # 提高阈值以获得更高质量的伪标签
```

或直接调用 Python 脚本：
```bash
python generate_pseudo_labels.py \
    --weights output/model_final.pth \
    --dataset-name city_trainT \
    --output pseudo_labels/city_trainT_thr08_coco.json \
    --score-threshold 0.8
```

### 伪标签质量评估标准
- ✅ **良好**: 平均置信度 > 0.75，类别分布相对均衡
- ⚠️ **中等**: 平均置信度 0.65-0.75，部分类别不平衡
- ❌ **较差**: 平均置信度 < 0.65，严重类别不平衡

### 可视化技巧
- 先分析整体统计，再可视化特定样本
- 对比不同阈值生成的伪标签质量
- 重点检查低置信度样本和稀有类别

---

## 🔧 故障排除

### 问题 1: 图像路径错误
**错误**: `✗ 图像不存在: /path/to/image.jpg`

**解决**:
- 检查 `--image-root` 参数是否正确
- 确认图像文件确实存在于该目录
- 伪标签对应的目标域图像应在 `datasets/cityscape/train_t/`

### 问题 2: 模型文件不存在
**错误**: `✗ ERROR: 文件不存在: output/model_final.pth`

**解决**:
- 确认已完成模型训练
- 或使用其他检查点: `export MODEL_PATH=output/model_0004999.pth`

### 问题 3: 依赖包缺失
**错误**: `ModuleNotFoundError: No module named 'matplotlib'`

**解决**:
```bash
pip install matplotlib pillow numpy
```

---

## 📝 参考

- 主训练脚本: `uda_train.py`
- 数据集注册: `register_cityscapes.py`
- 伪标签生成器: `generate_pseudo_labels.py`
- DA2OD 配置: `configs/da2od_config.yaml`

---

**更新日期**: 2025-11-11
**作者**: DA-Faster-RCNN Project
