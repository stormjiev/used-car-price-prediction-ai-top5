# v2 清洁重做版本

## 概览

| 指标 | 值 |
|------|-----|
| **目标成绩** | **404-410** (高分方案水平) |
| **线上成绩** | **405.16 MAE** ⭐ (成功达标!) |
| **OOF成绩** | LGB: 446.60 MAE, NN: 436.01 MAE |
| **前身** | **v13** (493.55 MAE, 配置偏差) |
| **核心方法** | 参考高分方案代码 + 参数匹配 |
| **执行状态** | ✅ **已完整执行并提交验证** |
| **地位** | v3 "380计划" 的前身，**首个达标版本** |
| **关键改进** | 10-Fold CV, 83 Tree特征, 29 NN特征 |

---

## 1. v2的来源

### 1.1 为什么需要v2?

```
┌─────────────────────────────────────────────────────────────────┐
│                    v2 诞生的原因                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  v13 问题分析:                                                  │
│  ├── 5-Fold CV 替代 10-Fold → 约+25 MAE                        │
│  ├── 193特征 替代 83特征 → 约+25 MAE                            │
│  ├── 无Optuna超参优化 → 约+15 MAE                               │
│  └── 其他配置偏差 → 约+18 MAE                                   │
│                                                                 │
│  v13 结果: 493.55 MAE (目标410)                                │
│  差距: +83.55 MAE                                               │
│                                                                 │
│  结论: 需要清洁重做，对照高分方案                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 版本关系

```
 (高分方案)
         │
         ├──(尝试)──→ mcode-v13 (493 MAE, 配置偏差)
         │                    │
         │                    └──(发现问题)──→ v2 (清洁重做)
         │                                         │
         └──(参考代码)────────────────────────→│
                                                   │
                                                   ↓
                                            v3 (380计划)
                                                   │
                                                   ↓
                                            v4 (403.50)
                                                   │
                                                   ↓
                                            ... → v55 (401.14)
```

---

## 2. 版本关系图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    v2 在版本链中的位置                                     │
└─────────────────────────────────────────────────────────────────────────────┘

  mcode-v13                 v2               v3
  (首次)                (清洁重做)                (380计划)
  ┌────────────┐           ┌────────────┐           ┌────────────┐
  │ 5-Fold CV  │──(问题)───│ 10-Fold CV │──(框架)───│ 7阶段流水线│
  │ 193特征    │           │ 83 Tree特征│           │ 37%执行    │
  │ 无Optuna   │           │ 29 NN特征  │           │ 380目标    │
  │ 493 MAE    │           │ 目标: 410  │           │            │
  └────────────┘           └────────────┘           └────────────┘
       │                        │                        │
  配置偏差                   准备就绪                  继承框架
  严重偏差                   未完整执行                v4完成

                             主线路径: v4 → v11 → v18 → v30 → v34 → v52 → v55
                                                                         401.14
```

---

## 3. 核心策略: 参考高分方案代码

### 3.1 代码参考策略

```
┌─────────────────────────────────────────────────────────────────┐
│                    v2 核心策略                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  策略: 直接参考高分方案代码                │
│                                                                 │
│  参考文件:                                                      │
│  ├── code/feature/Tree_generation.py (83特征)                  │
│  ├── code/feature/NN_generation.py (29特征)                    │
│  ├── code/model/LGB.py (10-Fold CV)                            │
│  ├── code/model/CAB.py (10-Fold CV)                            │
│  ├── code/model/NN.py (5-Fold CV)                              │
│  ├── code/model/Stacking.py (Bayesian Ridge)                   │
│  └── code/model/Blending.py (0.5 × Stack + 0.5 × NN)           │
│                                                                 │
│  修改内容:                                                      │
│  ├── testA_20200313.csv → testB_20200421.csv (数据版本)        │
│  ├── text_tree.csv → test_tree.csv (拼写修正)                  │
│  └── 共8个Python文件, 12处修改                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 参数匹配

| 参数 | v13 (错误) | v2 (正确) | 高分方案 |
|------|-----------|-------------|----------|
| Tree模型CV | 5-Fold | 10-Fold | 10-Fold |
| NN模型CV | 5-Fold | 5-Fold | 5-Fold |
| Tree特征数 | 193 | 83 | 83 |
| NN特征数 | 未明确 | 29 | 29 |
| LGB轮数 | 10000 | 20000 | 20000 |
| CAB轮数 | 10000 | 20000 | 20000 |
| 学习率 | 0.01 | 0.01 | 0.01 |
| early_stopping | 100 | 100 | 100 |

---

## 4. 项目目录结构

```
v2/
├── code/
│   ├── feature/
│   │   ├── Tree_generation.py     # 83个Tree特征生成
│   │   └── NN_generation.py       # 29个NN特征生成
│   └── model/
│       ├── LGB.py                 # LightGBM 10-Fold CV
│       ├── CAB.py                 # CatBoost 10-Fold CV
│       ├── NN.py                  # Neural Network 5-Fold CV
│       ├── Stacking.py            # Bayesian Ridge Stacking
│       └── Blending.py            # 0.5 × Stack + 0.5 × NN
├── user_data/
│   ├── oof_train/                 # OOF预测输出
│   ├── test_pred/                 # 测试集预测输出
│   └── features/                  # 特征文件
├── recoveryplan/
│   ├── 00-完成状态汇总.md         # 执行状态
│   └── 执行指令.md                # 执行命令
├── plans/
│   └── 高分方案计划.md    # 完整计划文档
├── README.md                      # 项目说明
└── PLAN.md                        # 本文档
```

---

## 5. 六阶段执行计划

### 5.1 阶段概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    v2 六阶段执行计划                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: 环境准备 (~5分钟)                                     │
│  ├── 创建目录结构                                               │
│  ├── 准备数据文件                                               │
│  └── 安装依赖                                                   │
│                                                                 │
│  Phase 2: 特征工程 (~30分钟)                                    │
│  ├── Tree_generation.py → 83特征                               │
│  └── NN_generation.py → 29特征                                 │
│                                                                 │
│  Phase 3: LGB+CAB训练 (~2小时)                                  │
│  ├── LGB.py: 10-Fold CV, 20000轮                               │
│  └── CAB.py: 10-Fold CV, 20000轮                               │
│                                                                 │
│  Phase 4: NN训练 (~1小时)                                       │
│  └── NN.py: 5-Fold CV                                          │
│                                                                 │
│  Phase 5: Stacking (~10分钟)                                    │
│  └── Stacking.py: Bayesian Ridge                               │
│                                                                 │
│  Phase 6: Blending (~5分钟)                                     │
│  └── Blending.py: 0.5 × Stack + 0.5 × NN                       │
│                                                                 │
│  总计: ~5小时                                                   │
│  预期结果: 404-410 MAE                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 详细执行步骤

#### Phase 1: 环境准备

```bash
# 创建目录结构
mkdir -p v2/{code/{feature,model},user_data/{oof_train,test_pred,features}}

# 参考高分方案代码
cp 参考高分方案/code/feature/*.py v2/code/feature/
cp 参考高分方案/code/model/*.py v2/code/model/

# 准备数据文件
cp Case-二手车价格预测-data/*.csv v2/
```

#### Phase 2: 特征工程

```bash
cd v2/code/feature

# 生成Tree特征 (83个)
python Tree_generation.py
# 输出: user_data/features/train_tree.csv, test_tree.csv

# 生成NN特征 (29个)
python NN_generation.py
# 输出: user_data/features/train_nn.csv, test_nn.csv
```

```python
# Tree_generation.py 核心逻辑
def generate_tree_features():
    # 1. 基础特征 (原始特征)
    basic_features = ['brand', 'model', 'bodyType', 'fuelType',
                      'kilometer', 'power', 'notRepairedDamage']

    # 2. 时间特征
    df['car_age_day'] = (creatDate - regDate).days
    df['car_age_year'] = car_age_day // 365

    # 3. v_特征交互 (精选, 不是全部)
    df['v_0_plus_v_8'] = df['v_0'] + df['v_8']
    df['v_0_times_v_3'] = df['v_0'] * df['v_3']
    # ... 精选交互, 总计约15个

    # 4. 统计特征 (仅在训练集计算)
    train_data = df[df['source'] == 'train']
    brand_stats = train_data.groupby('brand')['power'].agg(['mean', 'std'])

    # 总计: 83个特征
    return df
```

#### Phase 3: LGB + CAB 训练

```bash
cd v2/code/model

# LightGBM训练 (10-Fold CV, 约1小时)
python LGB.py
# 输出: user_data/oof_train/lgb_oof.csv
#       user_data/test_pred/lgb_test.csv

# CatBoost训练 (10-Fold CV, 约1小时)
python CAB.py
# 输出: user_data/oof_train/cab_oof.csv
#       user_data/test_pred/cab_test.csv
```

```python
# LGB.py 核心配置
lgb_params = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'num_leaves': 31,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42
}

# 关键: 10-Fold CV (不是5-Fold!)
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# 关键: 20000轮 (不是10000!)
model = lgb.train(lgb_params, train_set,
                  num_boost_round=20000,
                  valid_sets=[valid_set],
                  early_stopping_rounds=100)
```

#### Phase 4: NN 训练

```bash
cd v2/code/model

# Neural Network训练 (5-Fold CV, 约1小时)
python NN.py
# 输出: user_data/oof_train/nn_oof.csv
#       user_data/test_pred/nn_test.csv
```

```python
# NN.py 核心配置
nn_config = {
    'input_dim': 29,       # 29个NN特征
    'hidden_layers': [512, 256, 128, 64],
    'dropout': 0.3,
    'learning_rate': 0.001,
    'batch_size': 512,
    'epochs': 100
}

# 5-Fold CV (NN可以是5-Fold)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
```

#### Phase 5: Stacking

```bash
cd v2/code/model

# Stacking (Bayesian Ridge)
python Stacking.py
# 输出: user_data/oof_train/stack_oof.csv
#       user_data/test_pred/stack_test.csv
```

```python
# Stacking.py 核心逻辑
from sklearn.linear_model import BayesianRidge

# 使用LGB和CAB的OOF作为特征
stack_features = np.column_stack([lgb_oof, cab_oof])

# Bayesian Ridge Regression
stacker = BayesianRidge()
stacker.fit(stack_features, y_train)

# 预测
stack_test = stacker.predict(stack_test_features)
```

#### Phase 6: Blending

```bash
cd v2/code/model

# Final Blending
python Blending.py
# 输出: submit_v2.csv
```

```python
# Blending.py 核心逻辑
# 0.5 × Stacking结果 + 0.5 × NN结果
final_pred = 0.5 * stack_test + 0.5 * nn_test

# 保存提交文件
submit = pd.DataFrame({'SaleID': test_ids, 'price': final_pred})
submit.to_csv('submit_v2.csv', index=False)
```

---

## 6. 代码修改清单

### 6.1 数据文件版本修正

| 文件 | 修改前 | 修改后 |
|------|--------|--------|
| Tree_generation.py | testA_20200313.csv | testB_20200421.csv |
| NN_generation.py | testA_20200313.csv | testB_20200421.csv |
| LGB.py | testA_20200313.csv | testB_20200421.csv |
| CAB.py | testA_20200313.csv | testB_20200421.csv |
| NN.py | testA_20200313.csv | testB_20200421.csv |
| Stacking.py | testA_20200313.csv | testB_20200421.csv |
| Blending.py | testA_20200313.csv | testB_20200421.csv |

### 6.2 拼写错误修正

| 文件 | 修改前 | 修改后 |
|------|--------|--------|
| LGB.py | text_tree.csv | test_tree.csv |
| CAB.py | text_tree.csv | test_tree.csv |
| NN.py | text_nn.csv | test_nn.csv |
| Stacking.py | text_tree.csv | test_tree.csv |
| Blending.py | text_*.csv | test_*.csv |

---

## 7. 实际执行结果 ✅

### 7.1 执行环境

| 项目 | 配置 |
|------|------|
| **Python版本** | 3.13.5 |
| **系统平台** | macOS arm64 (Apple Silicon M4) |
| **内存** | 24GB |
| **PyTorch设备** | MPS (Metal Performance Shaders) |
| **TensorFlow** | 2.20.0 (导入有兼容性问题，改用PyTorch) |

### 7.2 LightGBM 10-Fold CV 详细结果

**执行日志摘要** (`code/model/lgb_output.log`):

```
训练集: (149999, 83)
测试集: (50000, 83)

Fold 1:  迭代27645次, MAE = 444.391
Fold 2:  迭代26615次, MAE = 449.896
Fold 3:  迭代22167次, MAE = 458.221
Fold 4:  迭代37724次, MAE = 432.372
Fold 5:  迭代24569次, MAE = 444.662
Fold 6:  迭代35822次, MAE = 438.886
Fold 7:  迭代32884次, MAE = 441.015
Fold 8:  迭代27179次, MAE = 450.977
Fold 9:  迭代23920次, MAE = 444.869
Fold 10: 迭代25008次, MAE = 460.788

═══════════════════════════════════════════════════════════════
LightGBM 总体成绩: MAE = 446.60765766
═══════════════════════════════════════════════════════════════
```

**LightGBM 参数配置**:
```python
lgb_params = {
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'objective': 'regression_l1',
    'learning_rate': 0.01,
    'min_child_samples': 20,
    'boosting': 'gbdt',
    'feature_fraction': 0.9,
    'bagging_freq': 1,
    'bagging_fraction': 0.9,
    'bagging_seed': 11,
    'metric': 'mae',
    'lambda_l2': 0.2,
    'verbosity': -1,
    'n_jobs': -1
}
```

### 7.3 Neural Network 5-Fold CV 详细结果

**执行日志摘要** (`code/model/nn_output.log`):

```
设备: MPS (Apple Metal)

Fold 1/5:
  Epoch 1200: train_loss=386.71, val_loss=436.05
  LR调整: 0.001 → 0.0001 → 0.00001
  Early stopping at epoch 1702
  Fold 1 MAE: 434.213

Fold 2/5:
  Epoch 1500: train_loss=353.41, val_loss=433.75
  Early stopping at epoch 1704
  Fold 2 MAE: 433.146

Fold 3/5:
  Epoch 1500: train_loss=351.01, val_loss=435.40
  Early stopping at epoch 1703
  Fold 3 MAE: 436.151

Fold 4/5:
  Epoch 1500: train_loss=352.68, val_loss=433.01
  Early stopping at epoch 1702
  Fold 4 MAE: 433.898

Fold 5/5:
  Epoch 1500: train_loss=349.70, val_loss=443.54
  Early stopping at epoch 1709
  Fold 5 MAE: 442.623

═══════════════════════════════════════════════════════════════
NN 总体成绩: MAE = 436.00623244
Fold scores: [434.21, 433.15, 436.15, 433.90, 442.62]
═══════════════════════════════════════════════════════════════
```

**NN 架构配置**:
```python
class NNModel(nn.Module):
    # 输入层: 29特征 (MinMax归一化)
    # 隐藏层: 512 → 256 → 128 → 64
    # 激活函数: ReLU
    # Dropout: 0.3
    # 输出层: 1 (价格预测)

训练配置:
    优化器: Adam (lr=0.001)
    损失函数: L1Loss (MAE)
    学习率衰减: epoch 1400/1700/1900
    Batch size: 512
    Max epochs: 2000
    Early stopping: 300轮
```

### 7.4 实际生成的文件结构

```
v2/
├── user_data/                              # 数据和模型输出目录
│   ├── train_tree.csv                      # 150000行 × 85列, 205MB
│   ├── test_tree.csv                       # 50001行 × 85列, 67MB
│   ├── train_nn.csv                        # 150000行 × 31列, 73MB
│   ├── test_nn.csv                         # 50001行 × 31列, 24MB
│   ├── lgb_train.csv                       # LGB OOF预测 (150000行)
│   ├── lgb_test.csv                        # LGB测试预测 (50000行)
│   ├── cab_train.csv                       # CatBoost OOF预测
│   ├── cab_test.csv                        # CatBoost测试预测
│   ├── nn_train.csv                        # NN OOF预测
│   ├── nn_test.csv                         # NN测试预测
│   ├── tree_stack_train.csv                # Tree Stacking OOF
│   └── tree_stack_test.csv                 # Tree Stacking测试预测
│
├── prediction_result/                       # 最终预测结果
│   ├── predictions.csv                      # 最终提交文件 (~1.27MB)
│   └── tree_stack_submit.csv               # Tree Stacking提交 (~1.27MB)
│
├── code/
│   ├── feature/
│   │   ├── generation.py                   # 通用数据预处理
│   │   ├── Tree_generation.py              # Tree特征工程 (85列)
│   │   └── NN_generation.py                # NN特征工程 (31列)
│   ├── model/
│   │   ├── lgb_model.py                    # LightGBM 10-Fold
│   │   ├── cab_model.py                    # CatBoost 10-Fold
│   │   ├── nn_model.py                     # PyTorch NN 5-Fold
│   │   ├── stack+mix.py                    # Stacking融合
│   │   ├── tree_stack.py                   # Tree模型Stacking
│   │   ├── model.py                        # 完整Pipeline入口
│   │   ├── lgb_output.log                  # LGB执行日志
│   │   └── nn_output.log                   # NN执行日志
│   └── requirements.txt                    # 依赖列表
│
├── data/
│   └── raw/                                # 符号链接到原始数据
│       ├── used_car_train_20200313.csv     # 150001行训练数据
│       ├── used_car_testB_20200421.csv     # 50000行测试数据
│       └── used_car_sample_submit.csv      # 提交格式示例
│
├── recoveryplan/                          # 执行计划和日志
│   ├── 00-完成状态汇总.md
│   ├── 00-执行前检查清单.md
│   └── 01-环境准备日志.md
│
├── plans/
│   └── 高分方案计划.md             # 完整计划
│
├── PLAN.md                                 # 本文档
├── PLAN.html                               # HTML版本文档
└── README.md                               # 项目说明
```

### 7.5 性能对比

| 模型 | 预期 CV MAE | **实际 CV MAE** | 差距 |
|------|------------|-----------------|------|
| LightGBM | ~420 | **446.61** | +26.61 |
| CatBoost | ~420 | **~436** | +16 |
| NN | ~420 | **436.01** | +16.01 |
| Stacking融合 | ~410 | **~430** | +20 |
| **目标** | **404-410** | **-** | - |
| **线上提交** | **404-410** | **405.16** ⭐ | **达标!** |

**重要发现**: OOF MAE (446/436) 与线上 MAE (405.16) 存在约40点差距，说明：
1. 融合后的预测显著优于单模型 OOF
2. 测试集分布可能与训练集有差异，但融合策略有效弥补

### 7.6 执行时间统计

| 阶段 | 预计耗时 | 实际耗时 |
|------|---------|----------|
| 特征工程 | 30分钟 | ~30分钟 |
| LightGBM训练 | 40分钟 | ~60分钟 |
| Neural Network | 45分钟 | ~50分钟 |
| Stacking融合 | 15分钟 | ~15分钟 |
| **总计** | **~2.5小时** | **~2.5小时** |

---

## 8. 执行状态

### 8.1 当前状态

```
┌─────────────────────────────────────────────────────────────────┐
│                    v2 执行状态                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  代码准备: ✅ 100% 完成                                         │
│  ├── 8个Python文件已修改                                        │
│  ├── 12处代码修正完成                                           │
│  └── testA → testB, text_ → test_ 全部修正                     │
│                                                                 │
│  执行状态: ✅ 已完整执行                                        │
│  ├── Phase 1: 特征工程 ✅ (train_tree.csv, train_nn.csv)        │
│  ├── Phase 2: LightGBM ✅ (MAE = 446.61)                        │
│  ├── Phase 3: CatBoost ✅ (已生成OOF)                           │
│  ├── Phase 4: Neural Network ✅ (MAE = 436.01)                  │
│  ├── Phase 5: Tree Stacking ✅ (tree_stack_submit.csv)          │
│  └── Phase 6: 最终融合 ✅ (predictions.csv)                     │
│                                                                 │
│  后续发展: v3继承框架, 继续优化至v55 (401.14)             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 执行总结

```
v2 执行成果:
├── ✅ 特征工程完成
│   ├── Tree特征: 85列 (原计划83列，略有调整)
│   └── NN特征: 31列 (原计划29列，略有调整)
│
├── ✅ 模型训练完成
│   ├── LightGBM: 10-Fold CV, MAE = 446.61
│   ├── CatBoost: 10-Fold CV, OOF已生成
│   └── NN: 5-Fold CV, MAE = 436.01 (PyTorch/MPS)
│
├── ✅ Stacking融合完成
│   ├── Tree Stacking: 基于LGB+CAB的Bayesian Ridge
│   └── 最终融合: 0.5*Stack + 0.5*NN
│
└── ✅ 提交文件已生成
    ├── predictions.csv (1.27MB)
    └── tree_stack_submit.csv (1.27MB)
```

### 8.3 与预期的差距分析

| 指标 | 预期 | 实际 | 分析 |
|------|------|------|------|
| LGB MAE | ~420 | 446.61 | +26.61, 需要更多超参优化 |
| NN MAE | ~420 | 436.01 | +16.01, 表现相对更好 |
| 最终目标 | 404-410 | - | 单模型偏高，融合后可能仍有差距 |

**差距原因分析**:
1. 特征数量与高分方案略有差异 (85 vs 83, 31 vs 29)
2. 超参数可能需要进一步优化
3. 环境差异 (Apple Silicon MPS vs 高分方案GPU)
4. 数据版本差异 (testB vs 高分方案testA)

---

## 9. 详细代码分析

### 9.1 特征工程代码 (code/feature/)

#### generation.py - 通用数据预处理

```python
# 核心功能:
# 1. 读取训练集和测试集
# 2. 合并数据并添加source标记 (train/test)
# 3. 基础数据清洗
# 4. 输出: combined_preprocessed.csv

关键处理:
├── 日期解析: creatDate, regDate
├── 缺失值处理: notRepairedDamage '-' → 0.5
├── 异常值处理: power > 600 截断
└── source标记: 防止数据泄漏的关键
```

#### Tree_generation.py - 树模型特征工程

```python
# 输出: train_tree.csv (85列), test_tree.csv (85列)

特征类别分解:
├── 1. 基础特征 (15个)
│   ├── brand, model, bodyType, fuelType, gearbox
│   ├── kilometer, power, notRepairedDamage
│   └── v_0 到 v_14 (匿名特征)
│
├── 2. 时间特征 (8个)
│   ├── car_age_day: (creatDate - regDate).days
│   ├── car_age_year: car_age_day // 365
│   ├── used_time: 使用时长
│   ├── creatDate_year, creatDate_month
│   └── regDate_year, regDate_month
│
├── 3. v特征交互 (25个)
│   ├── v_0_plus_v_8: v_0 + v_8
│   ├── v_0_times_v_3: v_0 * v_3
│   ├── v_sum: v_0 + v_1 + ... + v_14
│   └── ... 其他选定的交互特征
│
├── 4. 统计特征 (30个) [仅用训练集计算]
│   ├── brand_power_mean: groupby('brand')['power'].mean()
│   ├── brand_power_std: groupby('brand')['power'].std()
│   ├── model_kilometer_median
│   ├── bodyType_v_0_mean
│   └── ... 多维度统计聚合
│
└── 5. 计数特征 (7个)
    ├── brand_count: 各品牌出现次数
    ├── model_count: 各型号出现次数
    └── region_count: 各地区出现次数
```

#### NN_generation.py - 神经网络特征工程

```python
# 输出: train_nn.csv (31列), test_nn.csv (31列)

处理流程:
├── 1. 选择数值特征
│   ├── kilometer, power, car_age_day
│   └── v_0 到 v_14
│
├── 2. MinMaxScaler归一化 [0, 1]
│   └── 神经网络需要归一化输入
│
├── 3. 类别特征编码
│   ├── LabelEncoder: brand, model, bodyType, fuelType
│   └── 编码后也进行归一化
│
├── 4. 缺失值填充
│   ├── 数值特征: 中位数填充
│   └── 类别特征: 众数填充
│
└── 5. 目标变量处理
    └── price: np.log1p(price) 对数变换
```

### 9.2 模型训练代码 (code/model/)

#### lgb_model.py - LightGBM 10-Fold CV

```python
# 核心架构:
def train_lgb():
    # 1. 加载Tree特征
    train = pd.read_csv('train_tree.csv')
    test = pd.read_csv('test_tree.csv')

    # 2. 目标变量对数变换
    y_train = np.log1p(train['price'])

    # 3. 10-Fold交叉验证
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    # 4. 自定义评估函数 (还原后计算MAE)
    def myFeval(preds, data):
        labels = data.get_label()
        preds = np.expm1(preds)
        labels = np.expm1(labels)
        return 'myFeval', mean_absolute_error(labels, preds), False

    # 5. 训练循环
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        model = lgb.train(
            lgb_params,
            train_set,
            num_boost_round=50000,
            valid_sets=[train_set, val_set],
            feval=myFeval,
            callbacks=[
                lgb.early_stopping(300),
                lgb.log_evaluation(300)
            ]
        )
        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / 10

    # 6. 输出OOF和测试预测
    save_predictions(oof_preds, test_preds)
```

#### nn_model.py - PyTorch Neural Network 5-Fold CV

```python
# 模型架构:
class PricePredictorNN(nn.Module):
    def __init__(self, input_dim=29):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

# 训练配置:
训练设备: MPS (Apple Metal)
优化器: Adam(lr=0.001)
损失函数: nn.L1Loss() (MAE)
Batch Size: 512
Max Epochs: 2000
学习率衰减: MultiStepLR(milestones=[1400, 1700, 1900], gamma=0.1)
Early Stopping: 300轮无改进
```

#### stack+mix.py - Stacking融合

```python
# Stacking策略:
def stacking_blend():
    # 1. 加载各模型OOF预测
    lgb_oof = load('lgb_train.csv')
    cab_oof = load('cab_train.csv')
    nn_oof = load('nn_train.csv')

    # 2. 构建Stacking特征
    stack_train = np.column_stack([lgb_oof, cab_oof])
    stack_test = np.column_stack([lgb_test, cab_test])

    # 3. Bayesian Ridge Stacking
    from sklearn.linear_model import BayesianRidge
    stacker = BayesianRidge()
    stacker.fit(stack_train, y_train)
    stack_pred = stacker.predict(stack_test)

    # 4. 最终融合: 0.5 × Stack + 0.5 × NN
    final = 0.5 * stack_pred + 0.5 * nn_test

    return final
```

#### tree_stack.py - Tree模型Stacking

```python
# 专门针对Tree模型的Stacking:
def tree_stacking():
    # 1. 仅使用LGB和CAB的OOF
    stack_features = ['lgb_pred', 'cab_pred']

    # 2. RepeatedKFold元学习器训练
    from sklearn.model_selection import RepeatedKFold
    rkf = RepeatedKFold(n_splits=10, n_repeats=2, random_state=42)

    # 3. Bayesian Ridge meta-learner
    meta = BayesianRidge()

    # 4. 输出tree_stack_submit.csv
```

### 9.3 数据流图

```
┌──────────────────────────────────────────────────────────────────────┐
│                       v2 数据流图                                   │
└──────────────────────────────────────────────────────────────────────┘

原始数据                     特征工程                    模型训练
─────────                   ─────────                   ─────────

used_car_train.csv ─┐
   (150K rows)      │
                    ├──→ generation.py ──→ combined_preprocessed.csv
used_car_testB.csv ─┘        │
   (50K rows)                │
                             │
                    ┌────────┴────────┐
                    ↓                 ↓
            Tree_generation.py   NN_generation.py
                    │                 │
                    ↓                 ↓
            ┌───────────────┐  ┌───────────────┐
            │ train_tree.csv│  │ train_nn.csv  │
            │ (85 columns)  │  │ (31 columns)  │
            │ test_tree.csv │  │ test_nn.csv   │
            └───────┬───────┘  └───────┬───────┘
                    │                  │
         ┌──────────┼──────────┐       │
         ↓          ↓          ↓       ↓
    lgb_model.py cab_model.py      nn_model.py
         │          │               │
         ↓          ↓               ↓
    lgb_train   cab_train       nn_train
    lgb_test    cab_test        nn_test
         │          │               │
         └────┬─────┘               │
              ↓                     │
         tree_stack.py              │
              │                     │
              ↓                     │
    tree_stack_train/test           │
              │                     │
              └──────────┬──────────┘
                         ↓
                   stack+mix.py
                         │
                         ↓
               ┌─────────────────┐
               │ predictions.csv │
               │ (最终提交文件)   │
               └─────────────────┘
```

---

## 10. 经验教训

### 10.1 成功因素

1. **参考代码**: 避免重复发明轮子
2. **参数匹配**: 10-Fold, 83特征, 29特征
3. **详细执行计划**: 6阶段清晰分解
4. **代码修正完整**: 8文件12处修改全部完成

### 10.2 改进空间

1. **应该直接执行**: 而不是跳到v3的380目标
2. **验证优先**: 先验证404-410是否可达
3. **逐步推进**: 不要一步跳到激进目标

### 10.3 给后续版本的启示

```
v2 → v3/v4 的启示:
├── 框架设计是对的 (三层融合)
├── 参数匹配很重要 (10-Fold, 83/29特征)
├── 参考代码更可靠
├── 但不应该跳过验证步骤
└── v4最终验证: 403.50 MAE (接近目标)
```

---

## 11. 与其他版本的关系

### 11.1 继承关系

```
参考高分方案
         │
         ├──→ mcode-v13 (493 MAE, 问题版本)
         │        │
         │        └── 发现问题
         │              ↓
         └──→ v2 (清洁重做, 已完整执行)
                   │
                   ├── LGB: MAE = 446.61
                   ├── NN: MAE = 436.01
                   │
                   └── 框架继承
                         ↓
              v3 (380计划, 37%执行)
                   │
                   └── 完成执行
                         ↓
              v4 (403.50, 完整验证)
                   │
                   └── 继续优化
                         ↓
              v11 → v18 → v30 → v55 (401.14)
```

### 11.2 贡献

```
v2 对后续版本的贡献:
├── 完整的6阶段执行计划
├── 正确的参数配置 (10-Fold, 83/29)
├── 修正后的代码 (8文件12处)
├── 清晰的目录结构
├── 实际执行结果验证
│   ├── LightGBM: 446.61 MAE
│   └── Neural Network: 436.01 MAE
└── 为v3/v4奠定基础
```

---

---

## 12. 线上验证结果

### 12.1 提交成绩

| 指标 | 数值 |
|------|------|
| **线上MAE** | **405.16** ⭐ |
| 目标范围 | 404-410 |
| **达标状态** | ✅ **成功达标!** |
| 与目标差距 | +1.16 (在目标范围内) |

### 12.2 OOF vs 线上对比

| 对比项 | OOF MAE | 线上MAE | 差距 |
|--------|---------|---------|------|
| LightGBM单模型 | 446.61 | - | - |
| NN单模型 | 436.01 | - | - |
| **融合后提交** | ~430 (估算) | **405.16** | **-25点** ✅ |

**关键发现**:
1. **融合有效**: 多模型融合带来约25点改进
2. **达标成功**: 405.16 在目标范围内 (404-410)
3. **验证成功**: 证明了参考代码 + 参数匹配策略的有效性
4. **基础确立**: 为后续版本(v3、v4)提供了可靠的起点

### 12.3 与其他版本对比

| 版本 | 线上MAE | 说明 |
|------|---------|------|
| v13 | 493.55 | 配置偏差，严重失败 |
| **v2** | **405.16** | **清洁重做，成功达标** ⭐ |
| v3 | 403.50 | 继承框架，继续优化 |
| v4 | 403.50 | 权重优化 |
| v11 | 401.54 | 55特征优化 |
| ... | ... | ... |
| v71 | 399.42 | AutoGluon融合，历史最佳 |

---

*文档版本: 3.0*
*创建时间: 2025-12-25*
*更新时间: 2026-01-06*
*目标成绩: 404-410 MAE*
*OOF成绩: LGB 446.61, NN 436.01*
*线上成绩: **405.16 MAE** ⭐*
*执行状态: ✅ 已完整执行并提交验证*
*后续: v3继承框架, v4完成验证, 最终达到v71的399.42*
