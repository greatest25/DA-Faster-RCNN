# 代码修改总结

## 修改日期
2025-11-11 17:10

## 修改目的
解决三个核心问题:
1. ✅ **评估Student模型**: 当前只评估Teacher(EMA),应该评估Student以反映训练效果
2. ✅ **权重加载方式**: 说明为什么使用`MODEL.WEIGHTS`而不是`model_weights`参数
3. ✅ **输出路径配置**: 修改训练输出目录,方便管理多次实验

## 具体修改

### 1. 修改 `da2od/trainer.py` (核心改动)

**文件备份**: `da2od/trainer.py.backup`

**修改内容** (lines 213-242):
```python
def build_hooks(self):
    ret = super(DA2ODTrainer, self).build_hooks()

    # ============= 修改: 同时评估Student和Teacher =============
    # 1. 始终评估Student模型 (训练的主要模型)
    def test_and_save_results_student():
         self._last_eval_results_student = self.test(self.cfg, self.model)
         return self._last_eval_results_student
    eval_hook_student = hooks.EvalHook(self.cfg.TEST.EVAL_PERIOD, test_and_save_results_student)
    if comm.is_main_process():
         ret.insert(-1, eval_hook_student)  # before PeriodicWriter
    else:
         ret.append(eval_hook_student)
    
    # 2. 如果启用EMA, 也评估Teacher模型 (对比Student vs Teacher)
    if self.cfg.EMA.ENABLED:
         def test_and_save_results_teacher():
              self._last_eval_results_teacher = self.test(self.cfg, self.ema.model)
              return self._last_eval_results_teacher
         eval_hook_teacher = hooks.EvalHook(self.cfg.TEST.EVAL_PERIOD, test_and_save_results_teacher)
         if comm.is_main_process():
              ret.insert(-1, eval_hook_teacher)
         else:
              ret.append(eval_hook_teacher)

    # BestCheckpointer保存Student的最佳模型
    if comm.is_main_process():
        ret.insert(-1, BestCheckpointer(
            self.cfg.TEST.EVAL_PERIOD, self.checkpointer,
            f"bbox/AP50", "max", 
            file_prefix=f"{self.cfg.DATASETS.TEST[0]}_student_best"  # ← 修改
        ))
```

**改动说明**:
- **之前**: 仅评估Teacher(EMA)模型
- **现在**: 同时评估Student和Teacher模型
- **原因**: Student才是训练的主体,应该评估其性能
- **好处**: 可以对比Student vs Teacher,判断EMA的提升效果

### 2. 创建新配置 `configs/da2od_student_eval.yaml`

**与`da2od_full.yaml`的区别**:
```yaml
# 唯一修改: 输出路径
OUTPUT_DIR: "./output_da2od_student"  # 之前: ./output_da2od_full
```

**使用方式**:
```bash
python uda_train_simple.py --config-file configs/da2od_student_eval.yaml
```

## 问题解答

### Q1: 为什么之前只评估Teacher(EMA)?

**理论依据**:
- Mean Teacher论文中,Teacher模型通常性能更好
- Teacher权重是Student的指数移动平均,更稳定
- 原始DA2OD框架设计中,Teacher是最终部署模型

**但是**:
- **训练目标是强化Student**: 应该评估Student的学习效果
- **对比公平性**: Baseline评估的是直接训练模型,对应DA2OD的Student
- **诊断价值**: 同时评估可以看出EMA的实际贡献

### Q2: 为什么不使用model_weights参数?

**历史问题**:
```python
TypeError: GeneralizedRCNN.__init__() got an unexpected keyword argument 'model_weights'
```

**原因分析**:
1. **DA-RCNN修改**: `detectron2-main-DA-RCNN-modified`添加了`model_weights`参数
   - 用于加载discriminator权重
   - 是针对DA-RCNN的特殊修改
   
2. **原始detectron2**: 标准版本没有这个参数
   - DA2OD基于标准detectron2
   - 使用`MODEL.WEIGHTS`配置项加载预训练权重
   
3. **当前解决方案**:
   - 恢复原始detectron2 (从zip备份)
   - COCO权重通过`MODEL.WEIGHTS`加载
   - Discriminator(img_align)从头训练

**配置方式**:
```yaml
MODEL:
  WEIGHTS: "/path/to/model_final_b275ba.pkl"  # ← 标准detectron2加载方式
```

### Q3: 如何修改输出路径?

**方法1: 配置文件** (推荐)
```yaml
OUTPUT_DIR: "./output_custom_name"
```

**方法2: 命令行覆盖**
```bash
python uda_train_simple.py \
    --config-file configs/da2od_full.yaml \
    OUTPUT_DIR ./output_experiment_v2
```

**方法3: 代码修改**
```python
# uda_train_simple.py
cfg.OUTPUT_DIR = "./output_new_path"
```

## 预期结果

### 训练输出 (output_da2od_student/)
```
output_da2od_student/
├── city_testT_student_best.pth  ← Student最佳模型
├── model_final.pth              ← 最终checkpoint
├── model_000*.pth               ← 中间checkpoints
├── metrics.json                 ← 训练指标
└── inference/                   ← 评估结果
```

### 日志中会看到两次评估
```
[iter 1000] Evaluating city_testT (Student model)
 Average Precision (AP) @[ IoU=0.50:0.95 ] = 0.XXX
 
[iter 1000] Evaluating city_testT (Teacher model)  
 Average Precision (AP) @[ IoU=0.50:0.95 ] = 0.YYY
```

### 对比分析
- **Student vs Baseline**: 判断域适应的效果
- **Student vs Teacher**: 判断EMA的贡献
- **预期**: Teacher ≥ Student > Baseline

## 使用建议

### 立即测试 (小规模验证)
```bash
# 100次迭代快速测试
python uda_train_simple.py \
    --config-file configs/da2od_student_eval.yaml \
    SOLVER.MAX_ITER 100 \
    OUTPUT_DIR ./output_test_student
```

### 完整训练
```bash
# 5000次迭代完整训练
cd /mnt/lyh/DA-FasterCNN/DA-Faster-RCNN
nohup python uda_train_simple.py \
    --config-file configs/da2od_student_eval.yaml \
    > logs/da2od_student_eval.log 2>&1 &
```

### 对比之前的结果
```bash
# 之前(Teacher): AP=28.2%, AP50=48.1%
# 现在看Student性能是否更接近Baseline(32.1%)或更低
```

## 回滚方式

如果需要恢复原始代码:
```bash
cp da2od/trainer.py.backup da2od/trainer.py
```

## 总结

✅ **修改完成**:
- Student模型现在会被评估
- Teacher模型也会评估(如果启用EMA)
- BestCheckpointer保存Student的最佳模型
- 输出路径可通过配置文件灵活修改

🎯 **核心改进**:
- 更准确地反映训练效果 (Student性能)
- 可以诊断EMA的作用 (Student vs Teacher)
- 与Baseline的对比更公平
