# DA2OD 评估模型说明

## 问题1: 为什么当前评估的是EMA模型而不是Student模型?

### 当前实现 (trainer.py lines 217-235)
```python
if self.cfg.EMA.ENABLED:
    def test_and_save_results_ema():
        self._last_eval_results = self.test(self.cfg, self.ema.model)  # ← 评估EMA (Teacher)
        return self._last_eval_results
    eval_hook = hooks.EvalHook(self.cfg.TEST.EVAL_PERIOD, test_and_save_results_ema)
```

### 为什么这样设计?

**理论依据 (Mean Teacher论文)**:
1. **Teacher模型更稳定**: EMA模型是Student权重的指数移动平均,变化更平滑
2. **噪声抗性更强**: 对伪标签噪声和梯度波动有更好的鲁棒性
3. **泛化性能更好**: Mean Teacher框架中,Teacher通常比Student泛化能力强

**DA2OD框架的设计**:
- Teacher (EMA) 生成伪标签指导Student
- Teacher自身也在持续学习 (通过EMA更新)
- 最终部署时使用Teacher模型

### 但是...你的观点也有道理!

**应该评估Student的理由**:
1. **最终目标**: 如果训练目标是强化Student,应该评估Student性能
2. **对比公平性**: Baseline评估的是直接训练的模型,对应DA2OD中的Student
3. **诊断价值**: 同时评估Student和Teacher可以看出EMA的提升效果

## 问题2: 不使用model_weights参数的原因

### 历史问题回顾
之前遇到的错误:
```python
TypeError: GeneralizedRCNN.__init__() got an unexpected keyword argument 'model_weights'
```

### 根本原因
- **DA-RCNN修改版**: detectron2-main-DA-RCNN-modified 添加了`model_weights`参数用于加载discriminator权重
- **原始detectron2**: 标准版本没有这个参数
- **DA2OD兼容性**: DA2OD基于标准detectron2设计,不需要`model_weights`

### 当前解决方案
1. **恢复原始detectron2**: 从backup zip恢复干净版本
2. **使用标准加载**: 通过`MODEL.WEIGHTS`配置项加载COCO预训练权重
3. **discriminator从头训练**: DA2OD的img_align判别器随机初始化

```yaml
MODEL:
  WEIGHTS: "/mnt/lyh/DA-FasterCNN/weights/.../model_final_b275ba.pkl"  # ← 标准方式
```

## 问题3: 修改训练输出路径

### 当前配置
```yaml
OUTPUT_DIR: "./output_da2od_full"
```

### 如何修改

**方法1: 修改配置文件 (推荐)**
```yaml
# configs/da2od_full.yaml
OUTPUT_DIR: "./output_da2od_student_eval"  # 改成你想要的路径
```

**方法2: 命令行覆盖**
```bash
python uda_train_simple.py \
    --config-file configs/da2od_full.yaml \
    OUTPUT_DIR ./output_custom_path
```

**方法3: 代码中修改**
```python
# uda_train_simple.py
cfg.OUTPUT_DIR = "./output_new_experiment"
```

## 建议的改进方案

### 方案A: 同时评估Student和Teacher (最佳)

**修改 trainer.py**:
```python
def build_hooks(self):
    ret = super().build_hooks()
    
    # 评估Student模型
    def test_student():
        self._last_eval_results_student = self.test(self.cfg, self.model)
        return self._last_eval_results_student
    ret.insert(-1, hooks.EvalHook(self.cfg.TEST.EVAL_PERIOD, test_student))
    
    # 如果启用EMA,也评估Teacher
    if self.cfg.EMA.ENABLED:
        def test_teacher():
            self._last_eval_results_teacher = self.test(self.cfg, self.ema.model)
            return self._last_eval_results_teacher
        ret.insert(-1, hooks.EvalHook(self.cfg.TEST.EVAL_PERIOD, test_teacher))
        
        # BestCheckpointer可以基于Student或Teacher的指标
        if comm.is_main_process():
            ret.insert(-1, BestCheckpointer(
                self.cfg.TEST.EVAL_PERIOD, self.checkpointer,
                "bbox/AP50", "max",  # 使用Student的AP50
                file_prefix=f"{self.cfg.DATASETS.TEST[0]}_student_best"
            ))
    
    return ret
```

### 方案B: 仅评估Student (满足你的需求)

**修改 trainer.py lines 217-221**:
```python
# 注释掉EMA评估,添加Student评估
if self.cfg.EMA.ENABLED:
    # 不评估EMA,只用于生成伪标签
    pass

# 始终评估Student模型 (无论是否启用EMA)
def test_student():
    self._last_eval_results = self.test(self.cfg, self.model)
    return self._last_eval_results
eval_hook = hooks.EvalHook(self.cfg.TEST.EVAL_PERIOD, test_student)
ret.insert(-1, eval_hook)
```

### 方案C: 配置化选择 (最灵活)

**添加配置项**:
```yaml
# configs/da2od_full.yaml
EMA:
  ENABLED: True
  ALPHA: 0.999
  EVAL_TEACHER: False  # ← 新增: 是否评估Teacher
  EVAL_STUDENT: True   # ← 新增: 是否评估Student
```

## 推荐行动

### 立即执行
1. **创建新配置**: `configs/da2od_student_eval.yaml`
   ```yaml
   _BASE_: "da2od_full.yaml"
   OUTPUT_DIR: "./output_da2od_student_eval"
   ```

2. **修改评估代码**: 按方案B修改`da2od/trainer.py`

3. **重新训练**: 
   ```bash
   python uda_train_simple.py --config-file configs/da2od_student_eval.yaml
   ```

### 对比验证
运行完成后对比:
- Student模型 vs Baseline (32.1%)
- Student vs Teacher (当前48.1%)
- 判断EMA的实际提升效果

这样能真实反映Student模型的学习效果!
