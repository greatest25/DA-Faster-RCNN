# DA2OD 项目文档

本目录包含DA2OD训练项目的所有文档资料。

## 📚 文档结构

```
docs/
├── README.md                          ← 本文件
├── DA2OD_TRAINING_RESULTS.md          ← 训练结果报告
├── DA2OD_PIPELINE_SUCCESS.md          ← 流程成功记录
├── CLEANUP_SUMMARY.md                 ← 清理总结
└── guides/                            ← 详细指南目录
    ├── BACKGROUND_TRAINING_GUIDE.md   ← 后台训练完整指南
    ├── MODIFICATIONS_SUMMARY.md       ← 代码修改总结
    └── EVALUATION_MODEL_EXPLANATION.md ← 评估模型说明
```

## 📖 快速导航

### 核心文档

#### 1. DA2OD_TRAINING_RESULTS.md
**完整训练结果报告**
- 训练配置参数
- 损失函数收敛曲线
- 模型性能评估 (AP, AP50, AP75)
- 各类别性能分析
- 性能对比 (Baseline vs DA2OD)
- 改进建议

**适用场景**: 查看训练最终结果,分析性能

#### 2. DA2OD_PIPELINE_SUCCESS.md
**DA2OD完整流程记录**
- 流程各阶段执行过程
- 遇到的问题和解决方案
- 关键技术细节
- 成功案例记录

**适用场景**: 了解整个项目执行流程

#### 3. CLEANUP_SUMMARY.md
**项目清理总结**
- 删除的文件列表
- 保留的核心文件
- 目录结构优化
- 释放的磁盘空间

**适用场景**: 了解项目整理情况

### 详细指南 (guides/)

#### 1. BACKGROUND_TRAINING_GUIDE.md ⭐ 重点
**后台训练完整指南**
- 上一轮训练数据处理
- 后台训练使用方法
- 监控命令速查
- 常见问题解答
- 完整工作流程

**适用场景**: 
- 第一次启动后台训练
- 需要监控训练状态
- 遇到训练问题

**快速开始**:
```bash
cd /mnt/lyh/DA-FasterCNN/DA-Faster-RCNN
bash scripts/run_training_background.sh
bash scripts/check_training.sh
```

#### 2. MODIFICATIONS_SUMMARY.md
**代码修改总结**
- 修改目的和动机
- 具体修改内容 (trainer.py)
- 三个核心问题解答
- 使用建议和测试方法

**适用场景**:
- 了解代码改了什么
- 为什么要评估Student模型
- 如何修改输出路径

#### 3. EVALUATION_MODEL_EXPLANATION.md
**评估模型详细说明**
- 为什么评估Teacher/Student
- model_weights参数说明
- 输出路径配置方法
- 改进方案对比

**适用场景**:
- 深入理解评估机制
- 了解Mean Teacher原理
- 权重加载方式说明

## 🎯 使用场景导航

### 场景1: 我想开始训练
1. 阅读 `guides/BACKGROUND_TRAINING_GUIDE.md`
2. 执行 `bash scripts/run_training_background.sh`
3. 使用 `bash scripts/check_training.sh` 监控

### 场景2: 我想了解训练结果
1. 查看 `DA2OD_TRAINING_RESULTS.md`
2. 检查日志: `tail -100 logs/da2od_student_training.log | grep AP`

### 场景3: 我遇到了问题
1. 查看 `guides/BACKGROUND_TRAINING_GUIDE.md` 的"常见问题"部分
2. 检查日志: `tail -100 logs/da2od_student_training.log`
3. 查看 `guides/MODIFICATIONS_SUMMARY.md` 了解代码改动

### 场景4: 我想理解代码为什么这样改
1. 阅读 `guides/MODIFICATIONS_SUMMARY.md`
2. 深入了解: `guides/EVALUATION_MODEL_EXPLANATION.md`

### 场景5: 我想复现整个流程
1. 阅读 `DA2OD_PIPELINE_SUCCESS.md` 了解完整流程
2. 按照 `guides/BACKGROUND_TRAINING_GUIDE.md` 执行

## 📊 训练结果对比

| 模型 | AP@50:95 | AP50 | 说明 |
|------|----------|------|------|
| Baseline (R50-FPN) | 32.1% | 55.0% | Source-only训练 |
| DA2OD Teacher | 28.2% | 48.1% | EMA模型(上一轮) |
| DA2OD Student | ? | ? | 本次训练目标 |

## 🔧 相关目录

- **scripts/** - 训练管理脚本
- **configs/** - 训练配置文件
- **logs/** - 训练日志
- **output_da2od_student/** - 新训练输出
- **output_da2od_full_teacher_only/** - 上一轮备份

## 📝 更新记录

- 2025-11-11: 初始文档结构创建
- 2025-11-11: 添加训练结果报告
- 2025-11-11: 添加后台训练指南
- 2025-11-11: 代码修改总结

## 💡 提示

- 所有文档都是Markdown格式,可直接在GitHub查看
- 使用 `grep` 命令快速搜索文档内容
- 定期备份训练结果和重要文档
