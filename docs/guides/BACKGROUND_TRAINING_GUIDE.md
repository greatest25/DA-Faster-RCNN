# 后台训练完整指南

## 📦 上一轮训练数据处理

### 已完成的操作
```bash
✅ 重命名: output_da2od_full → output_da2od_full_teacher_only
   - 保留了上一轮完整训练数据 (3.3GB)
   - 包含Teacher模型的所有checkpoint
   - 新训练不会覆盖这些数据
```

### 如果需要进一步节省空间

**选项1: 删除中间checkpoint** (释放2.4GB)
```bash
cd output_da2od_full_teacher_only
rm -f model_000*.pth  # 删除5个中间checkpoint
# 保留: city_testT_model_best.pth, model_final.pth
```

**选项2: 压缩归档** (3.3GB → 1.5GB)
```bash
tar czf output_da2od_full_teacher_only.tar.gz output_da2od_full_teacher_only/
rm -rf output_da2od_full_teacher_only/
```

## 🚀 后台训练使用方法

### 一键启动 (推荐)
```bash
cd /mnt/lyh/DA-FasterCNN/DA-Faster-RCNN
bash run_training_background.sh
```

**脚本会自动**:
1. ✅ 激活lyh_env环境
2. ✅ 检查是否已有训练在运行
3. ✅ 使用nohup在后台启动训练
4. ✅ 保存进程ID到文件
5. ✅ 创建日志文件
6. ✅ 检查启动是否成功

**启动后会显示**:
```
✅ 训练已启动!
   进程ID: 123456
   PID文件: /tmp/da2od_training.pid

📊 监控命令:
   实时日志: tail -f logs/da2od_student_training.log
   检查进程: ps -p 123456
   停止训练: kill 123456
   GPU使用: watch -n 1 nvidia-smi

🕐 预计训练时间: ~60分钟 (5000次迭代)

✅ 训练进程运行正常
   可以安全关闭SSH连接,训练会继续运行
```

### 检查训练状态
```bash
bash check_training.sh
```

**会显示**:
- ✅ 进程运行状态
- 📊 CPU/内存使用情况
- 📝 最新20行日志
- 🎯 训练进度 (当前迭代/总迭代)

### 停止训练
```bash
bash stop_training.sh
```

**优雅停止**: 先尝试正常终止,如果无响应则强制终止

## 📊 监控命令速查

### 实时查看日志
```bash
tail -f logs/da2od_student_training.log
```
按 `Ctrl+C` 退出监控

### 搜索特定内容
```bash
# 查看所有评估结果
grep "Average Precision" logs/da2od_student_training.log

# 查看loss变化
grep "total_loss" logs/da2od_student_training.log | tail -20

# 查看最后的AP
tail -100 logs/da2od_student_training.log | grep "AP"
```

### GPU使用监控
```bash
# 实时刷新 (每1秒)
watch -n 1 nvidia-smi

# 仅查看一次
nvidia-smi
```

### 进程监控
```bash
# 检查进程是否存在
PID=$(cat /tmp/da2od_training.pid)
ps -p $PID

# 查看详细信息
ps -p $PID -o pid,ppid,cmd,%cpu,%mem,etime
```

## 🔧 手动操作 (如果脚本不可用)

### 手动启动后台训练
```bash
cd /mnt/lyh/DA-FasterCNN/DA-Faster-RCNN
conda activate lyh_env

nohup python uda_train_simple.py \
    --config-file configs/da2od_student_eval.yaml \
    > logs/da2od_student_training.log 2>&1 &

echo $! > /tmp/da2od_training.pid
echo "进程ID: $(cat /tmp/da2od_training.pid)"
```

### 手动停止训练
```bash
PID=$(cat /tmp/da2od_training.pid)
kill $PID  # 正常终止

# 如果无响应
kill -9 $PID  # 强制终止
```

## 📁 输出文件位置

### 新训练输出
```
output_da2od_student/
├── city_testT_student_best.pth  ← Student最佳模型
├── model_final.pth              ← 最终模型
├── model_000*.pth               ← 每1000次迭代checkpoint
├── metrics.json                 ← 训练指标
├── events.out.tfevents.*        ← TensorBoard日志
└── inference/                   ← 评估结果
```

### 日志文件
```
logs/
└── da2od_student_training.log   ← 完整训练日志
```

### 上一轮备份
```
output_da2od_full_teacher_only/  ← 上一轮Teacher模型训练
```

## ⚠️ 常见问题

### Q1: SSH断开后训练会停止吗?
**A**: 不会! `nohup`确保进程在后台持续运行,即使SSH断开。

### Q2: 如何在断开连接后重新查看?
**A**: 重新SSH登录后执行:
```bash
bash check_training.sh
tail -f logs/da2od_student_training.log
```

### Q3: 训练意外停止怎么办?
**A**: 检查日志找原因:
```bash
tail -100 logs/da2od_student_training.log
```
常见原因:
- OOM (内存不足)
- CUDA错误
- 磁盘空间不足
- 代码错误

### Q4: 如何恢复中断的训练?
**A**: DA2OD会自动从最后的checkpoint恢复:
```bash
# 检查是否有checkpoint
ls -lh output_da2od_student/model_*.pth

# 重新启动(会自动加载last_checkpoint)
bash run_training_background.sh
```

### Q5: 多次启动会重复训练吗?
**A**: 不会! 脚本会检查是否已有训练在运行:
```
⚠️  检测到训练已在运行 (PID: 123456)
   如需停止: kill 123456
```

## 🎯 完整工作流程

### 1. 启动训练
```bash
cd /mnt/lyh/DA-FasterCNN/DA-Faster-RCNN
bash run_training_background.sh
```

### 2. 查看初始状态 (等待30秒)
```bash
sleep 30
bash check_training.sh
```

### 3. 安全断开SSH
```bash
# 确认训练正常后,可以直接关闭终端
exit
```

### 4. 稍后重新登录检查
```bash
ssh user@server
cd /mnt/lyh/DA-FasterCNN/DA-Faster-RCNN
bash check_training.sh
```

### 5. 训练完成后查看结果
```bash
# 查看最终AP
tail -100 logs/da2od_student_training.log | grep "Average Precision"

# 查看保存的模型
ls -lh output_da2od_student/*.pth
```

## 📈 预期训练时间

- **总迭代**: 5000次
- **每次迭代**: ~0.56秒
- **预计总时间**: 5000 × 0.56 = 2800秒 ≈ **47分钟**
- **加上评估**: 每1000次评估约1分钟,共5次 = 5分钟
- **总计**: 约 **52-60分钟**

## 🎉 训练完成标志

日志最后会显示:
```
[DATE TIME d2.engine.hooks]: Overall training speed: XXX iterations in X:XX:XX
[DATE TIME d2.engine.train_loop]: Total training time: X:XX:XX
```

模型文件会出现:
```
output_da2od_student/
├── city_testT_student_best.pth  ← 最佳Student模型
└── model_final.pth              ← 最终模型
```

## 📊 结果分析

训练完成后对比:
```bash
# Student vs Teacher vs Baseline
echo "Baseline (Source-only): AP=32.1%, AP50=55.0%"
echo "Teacher (之前训练): AP=28.2%, AP50=48.1%"
echo "Student (本次训练): 查看日志中的最终AP"

tail -100 logs/da2od_student_training.log | grep "Average Precision"
```

## 💡 提示

1. **定期检查**: 建议每10-15分钟检查一次训练状态
2. **GPU监控**: 确保GPU使用率在80-100%之间
3. **日志保存**: 训练结束后备份日志文件
4. **磁盘空间**: 确保有足够空间保存checkpoint (每个480MB)
5. **环境稳定**: 确认服务器不会自动重启

