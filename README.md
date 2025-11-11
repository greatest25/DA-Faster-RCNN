# DA-Faster-RCNN & DA2OD

域自适应目标检测项目 - Cityscapes Clear→Foggy

## 📋 项目概述

本项目实现了基于DA2OD(Differential Alignment for Domain Adaptive Object Detection)的域自适应目标检测,从Cityscapes清晰天气数据集适应到雾天场景。

### 核心框架
- **Baseline**: Faster R-CNN with ResNet50-FPN
- **DA方法**: DA2OD with Mean Teacher
- **数据集**: Cityscapes (clear→foggy, 8类)
- **框架**: Detectron2 0.6

## 🚀 快速开始

### 1. 启动训练
```bash
cd /mnt/lyh/DA-FasterCNN/DA-Faster-RCNN
bash scripts/run_training_background.sh
```

### 2. 检查状态
```bash
bash scripts/check_training.sh
```

### 3. 实时监控
```bash
tail -f logs/da2od_student_training.log
```

## 📁 项目结构

```
DA-Faster-RCNN/
├── README.md                    ← 本文件
├── uda_train_simple.py          ← 训练主程序
├── generate_pseudo_labels.py    ← 伪标签生成
├── evaluate_da2od.py           ← 模型评估
│
├── configs/                     ← 配置文件
│   ├── da2od_demo.yaml         ← 100次迭代demo
│   ├── da2od_full.yaml         ← 5000次完整训练(Teacher)
│   └── da2od_student_eval.yaml ← 5000次训练(Student评估)
│
├── scripts/                     ← 训练管理脚本
│   ├── README.md               ← 脚本说明
│   ├── run_training_background.sh  ← 启动后台训练
│   ├── check_training.sh           ← 检查训练状态
│   └── stop_training.sh            ← 停止训练
│
├── docs/                        ← 项目文档
│   ├── README.md               ← 文档索引
│   ├── DA2OD_TRAINING_RESULTS.md     ← 训练结果报告
│   ├── DA2OD_PIPELINE_SUCCESS.md     ← 流程记录
│   ├── CLEANUP_SUMMARY.md            ← 清理总结
│   └── guides/                       ← 详细指南
│       ├── BACKGROUND_TRAINING_GUIDE.md   ← 后台训练指南⭐
│       ├── MODIFICATIONS_SUMMARY.md       ← 代码修改说明
│       └── EVALUATION_MODEL_EXPLANATION.md ← 评估模型说明
│
├── da2od/                       ← DA2OD核心代码
│   ├── trainer.py              ← 训练器(已修改,同时评估Student/Teacher)
│   ├── model.py                ← 模型定义
│   ├── dataloader.py           ← 数据加载
│   └── ...
│
├── logs/                        ← 训练日志
├── output_da2od_student/        ← Student模型训练输出
├── output_da2od_full_teacher_only/ ← Teacher模型备份(上一轮)
├── output/resnet50_fpn_baseline/   ← Baseline模型
└── pseudo_labels/               ← 伪标签数据
```

## 📊 训练结果对比

| 模型 | AP@50:95 | AP50 | AP75 | 说明 |
|------|----------|------|------|------|
| R50-FPN Baseline | 32.1% | 55.0% | - | Source-only |
| DA2OD Teacher | 28.2% | 48.1% | 29.3% | EMA模型(上一轮) |
| DA2OD Student | ? | ? | ? | 待训练 |

## 🔧 环境配置

### 依赖
- Python 3.10
- PyTorch 2.0.1
- CUDA 11.8
- Detectron2 0.6
- Conda环境: lyh_env

### 激活环境
```bash
conda activate lyh_env
```

## 📖 详细文档

- **完整指南**: [docs/guides/BACKGROUND_TRAINING_GUIDE.md](docs/guides/BACKGROUND_TRAINING_GUIDE.md)
- **训练结果**: [docs/DA2OD_TRAINING_RESULTS.md](docs/DA2OD_TRAINING_RESULTS.md)
- **代码修改**: [docs/guides/MODIFICATIONS_SUMMARY.md](docs/guides/MODIFICATIONS_SUMMARY.md)
- **文档索引**: [docs/README.md](docs/README.md)

## 🎯 关键特性

### 1. Student & Teacher同时评估
- 修改了`da2od/trainer.py`,同时评估Student和Teacher模型
- Student是训练主体,更能反映域适应效果
- Teacher作为对比参考,判断EMA贡献

### 2. 后台训练管理
- 一键启动后台训练(nohup)
- SSH断开不影响训练
- 实时状态检查和进度追踪
- 智能PID进程管理

### 3. 完整流程
1. Baseline训练 (R50-FPN on clear data)
2. 伪标签生成 (Baseline预测foggy data)
3. DA2OD训练 (Source + Pseudo-labeled Target)
4. 评估对比 (Student vs Teacher vs Baseline)

## �� 常用命令

### 训练管理
```bash
# 启动训练
bash scripts/run_training_background.sh

# 检查状态
bash scripts/check_training.sh

# 停止训练
bash scripts/stop_training.sh

# 实时日志
tail -f logs/da2od_student_training.log
```

### 结果查看
```bash
# 查看AP结果
grep "Average Precision" logs/da2od_student_training.log

# 查看loss变化
grep "total_loss" logs/da2od_student_training.log | tail -20

# 查看保存的模型
ls -lh output_da2od_student/*.pth
```

### GPU监控
```bash
watch -n 1 nvidia-smi
```

## 🐛 常见问题

### Q1: SSH断开后训练会停止吗?
**A**: 不会!使用`nohup`后台运行,训练会持续进行。

### Q2: 如何查看训练进度?
**A**: 执行 `bash scripts/check_training.sh`

### Q3: 训练出错怎么办?
**A**: 查看日志 `tail -100 logs/da2od_student_training.log`

### Q4: 如何修改输出路径?
**A**: 编辑配置文件中的`OUTPUT_DIR`参数

详见: [docs/guides/BACKGROUND_TRAINING_GUIDE.md](docs/guides/BACKGROUND_TRAINING_GUIDE.md)

## �� 更新日志

### 2025-11-11
- ✅ 完成DA2OD完整流程(Teacher模型)
- ✅ 修改trainer.py支持Student评估
- ✅ 创建后台训练管理脚本
- ✅ 整理归档项目文档和脚本
- ✅ 上一轮训练数据备份

### 待完成
- ⏳ Student模型训练(5000次迭代)
- ⏳ Student vs Teacher性能对比
- ⏳ 伪标签质量分析和优化

## 📚 参考文献

- DA2OD: Differential Alignment for Domain Adaptive Object Detection
- Mean Teacher: Mean teachers are better role models
- Faster R-CNN: Towards Real-Time Object Detection
- Detectron2: A PyTorch-based modular object detection library

## 📧 联系方式

项目路径: `/mnt/lyh/DA-FasterCNN/DA-Faster-RCNN`

---

**准备就绪! 随时可以开始训练** 🚀
