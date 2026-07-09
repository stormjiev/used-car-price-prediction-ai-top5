# 高分方案计划 (方案版本)

**目标**: 在v2中参考高分方案，达成404-410 MAE的目标
**环境**: Python 3.13.5, 苹果Silicon M4 (arm64), 无GPU, 24GB内存
**数据**: `/Case-二手车价格预测-data/` (统一数据源)
**新建项目**: `v2/` (完整新项目，保留mcode-v13不动)
**文档输出**: `v2/recoveryplan/` (完整的执行日志、进度、分数对标、故障记录)
**高分方案**: `参考高分方案/` (代码参考)

---

## 关键决策

✅ **采用方案A**: 新建 `v2/` 作为清洁的项目目录
- 在 `v2/` 中从零开始清洁实现
- 便于对比和版本管理

✅ **文档策略**: `v2/recoveryplan/` 包含：
- 完整的执行日志和进度跟踪
- 最终分数对标和总结报告
- 故障排查记录（如有）
- 参数对照表和配置验证

✅ **计划存储位置**: `v2/plans/` 目录

---

## Phase 1: 环境准备与代码清理

### 1.1 环境检查与依赖安装

**当前已有**:
- Python 3.13.5 ✓
- pandas 2.2.3 ✓
- numpy 2.2.5 ✓
- scikit-learn ✓
- lightgbm 4.6.0 ✓
- catboost 1.2.8 ✓
- torch 2.9.1 ✓

**需要安装 - TensorFlow (苹果Silicon M4专用版本)**:
```bash
# 方法1: 使用conda (推荐用于苹果Silicon)
conda create -n tf-apple python=3.11
conda activate tf-apple
conda install -c apple tensorflow-deps
pip install tensorflow-macos tensorflow-metal

# 方法2: pip直接安装 (适用于M系列芯片)
pip install tensorflow-macos tensorflow-metal

# 方法3: 如果上述失败，使用通用版本
pip install tensorflow==2.13.0

# 方法4: 最后备选 - 使用PyTorch替代NN模块
pip install torch torchvision torchaudio
```

**环境适配**:
- 无CUDA/GPU支持 → 所有模型使用CPU模式
- 苹果Silicon M4 (arm64) → 充分利用多核并行 (CPU线程数调整)
- 内存24GB → 足够处理600MB特征数据
- Python 3.13.5 → 需要与TensorFlow版本兼容

### 1.2 新建v2项目结构

```
v2/                           # ← 新建：干净的高分方案项目
├── plans/                             # ← 计划和指南文档
│   └── 高分方案计划.md
├── code/                              # ← 参考自高分方案/code/
│   ├── feature/
│   │   ├── generation.py              # 通用预处理
│   │   ├── Tree_generation.py         # 树特征工程 (83特征)
│   │   └── NN_generation.py           # NN特征工程 (29特征)
│   ├── model/
│   │   ├── model.py                   # 完整Pipeline (推荐入口)
│   │   ├── lgb_model.py               # LGB 10-Fold训练
│   │   ├── cab_model.py               # CAB 10-Fold训练
│   │   ├── nn_model.py                # NN 5-Fold训练
│   │   └── stack+mix.py               # Stacking + Blending融合
│   └── requirements.txt               # 依赖
├── data/
│   ├── raw/                           # ← 符号链接到Case-二手车价格预测-data/
│   │   ├── used_car_train_20200313.csv
│   │   ├── used_car_testB_20200421.csv
│   │   └── used_car_sample_submit.csv
│   └── processed/                     # 特征、OOF、预测临时存储
├── results/
│   └── submissions/
│       └── predictions.csv            # 最终提交文件
├── user_data/                         # ← 模型输出临时目录 (由代码生成)
├── recoveryplan/                    # ← 完整的执行日志和文档
│   ├── 00-计划概览.md
│   ├── 01-环境准备日志.md
│   ├── 02-特征工程进度.md
│   ├── 03-模型训练进度.md
│   ├── 04-融合结果.md
│   ├── 05-参数对照表.md
│   └── 06-最终报告.md
└── README.md                          # 项目说明
```

---

## Phase 2: 代码与参数对照

### 2.1 从参考高分方案迁移代码到v2

**步骤**:
1. 迁移 `参考高分方案/code/` → `v2/code/`
2. 创建数据软链接: `v2/data/raw/` → `/Case-二手车价格预测-data/`
3. 创建 `v2/recoveryplan/` 目录用于完整文档
4. 创建 `v2/data/processed/` 目录存储中间数据

### 2.2 参数完全一致性检查

**树特征工程** (Tree_generation.py):
- 目标特征数: 83个
- 统计函数: mean, median, std, min, max, sum, skew, kurt, mad
- 分组字段: brand, model, bodyType, fuelType, kilometer
- 相关性过滤: Spearman > 0.95删除
- 防泄漏: source=='train'分离

**NN特征工程** (NN_generation.py):
- 目标特征数: 29个
- 归一化: MinMaxScaler [0,1]
- 编码: Label Encoding
- 缺失值处理: notRepairedDamage '-' → 0.5

**LightGBM** (lgb_model.py):
- learning_rate: 0.02
- num_leaves: 31
- lambda_l2: 2
- objective: regression_l1 (MAE)
- CV: 10-Fold
- Early Stopping: 300轮
- 预期单模型: ~420 MAE

**CatBoost** (cab_model.py):
- learning_rate: 0.02
- depth: 6
- reg_lambda: 3
- loss_function: MAE
- CV: 10-Fold
- Early Stopping: 300轮
- 预期单模型: ~420 MAE

**Neural Network** (nn_model.py):
- 架构: 512-256-128-64-1 (ReLU)
- 损失: MAE
- 优化器: Adam
- L2正则: 0.02
- Batch: 512
- Epochs: 2000
- 学习率衰减: epoch 1400/1700/1900
- CV: 5-Fold
- Early Stopping: 300轮
- 预期单模型: ~420 MAE

**Stacking** (stack+mix.py):
- Meta-learner: Bayesian Ridge Regression
- 输入: LGB OOF + CAB OOF
- CV: RepeatedKFold(n_splits=10, n_repeats=2) = 20折
- 预期: ~415 MAE

**最终融合**:
- 公式: Final = 0.5 * Stack_Pred + 0.5 * NN_Pred
- 输出格式: CSV (SaleID, price)
- 预期: 404-410 MAE

---

## Phase 3: 分阶段执行流程

### 3.1 Stage 1: 数据预处理 (30分钟)

```bash
cd v2/code/feature
python generation.py
```

**验证**:
- 输出文件: `user_data/combined_preprocessed.csv` (存在且包含所有150K训练样本)
- 特征数: 应为29-31列 (包括source和price)
- 完整性: 无NaN值

### 3.2 Stage 2: 特征工程 (双轨并行，共45分钟)

#### 2A: 树模型特征工程 (15分钟)

```bash
cd v2/code/feature
python Tree_generation.py
```

**验证**:
- `user_data/train_tree.csv`: 150K行 × 83列
- `user_data/test_tree.csv`: 50K行 × 83列

#### 2B: NN特征工程 (3分钟)

```bash
cd v2/code/feature
python NN_generation.py
```

**验证**:
- `user_data/train_nn.csv`: 150K行 × 29列
- `user_data/test_nn.csv`: 50K行 × 29列
- 数值范围: [0, 1] (MinMax归一化)

### 3.3 Stage 3: 基础模型训练 (2.5小时，可并行)

#### 3A: LightGBM 10-Fold CV (40分钟)

```bash
cd v2/code/model
python lgb_model.py
```

**输出**: `user_data/lgb_train.csv`, `user_data/lgb_test.csv`
**预期MAE**: 420-430

#### 3B: CatBoost 10-Fold CV (50分钟)

```bash
cd v2/code/model
python cab_model.py
```

**输出**: `user_data/cab_train.csv`, `user_data/cab_test.csv`
**预期MAE**: 420-430

#### 3C: Neural Network 5-Fold CV (45分钟)

```bash
cd v2/code/model
python nn_model.py
```

**输出**: `user_data/nn_train.csv`, `user_data/nn_test.csv`
**预期MAE**: 420-430

### 3.4 Stage 4: 模型融合 (15分钟)

```bash
cd v2/code/model
python stack+mix.py
```

**输出**: `user_data/predictions.csv` (最终提交文件)
**预期MAE**: 404-410 (线下估计)

---

## Phase 4: 验证与调试策略

### 4.1 每个阶段的验收标准

| 阶段 | 验收标准 | 异常处理 |
|------|---------|---------|
| 预处理 | 150K行无NaN | 检查数据源，重新预处理 |
| 树特征 | 83列无NaN | 检查生成逻辑，对比高分方案 |
| NN特征 | 29列[0,1]范围 | 检查归一化逻辑 |
| LGB | MAE 420±20 | 检查超参、数据、CV折分 |
| CAB | MAE 420±20 | 同上 |
| NN | MAE 420±30 | 检查TF版本、学习率、epoch |
| Stacking | MAE 410-420 | 检查OOF质量、meta-learner |
| 最终 | MAE 404-415 | 检查融合权重、数值范围 |

### 4.2 出错时的微调策略

**如果单模型分数偏低** (>450 MAE):
1. 检查特征数是否正确 (树83, NN29)
2. 检查数据预处理逻辑
3. 检查超参是否一致
4. 验证10-Fold CV实现
5. 检查防泄漏 (source标记)

**如果NN无法训练** (TensorFlow版本问题):
1. 检查TensorFlow是否正确安装: `python -c "import tensorflow; print(tensorflow.__version__)"`
2. 如果失败，尝试PyTorch替代方案
3. 检查兼容性: Python 3.13.5需要TensorFlow >= 2.13

**如果融合效果差** (MAE不下降):
1. 检查OOF预测的数值范围
2. 验证log/expm1变换
3. 调整融合权重 (0.5/0.5 → 0.4/0.6 或 0.6/0.4)
4. 检查Bayesian Ridge是否过拟合

---

## Phase 5: 文档管理

**recoveryplan/ 目录结构**:

```
recoveryplan/
├── 00-计划概览.md              # 本计划文档的摘要
├── 01-环境准备日志.md          # 依赖安装、代码清理、目录结构
├── 02-特征工程进度.md          # 树特征、NN特征的生成日志
├── 03-模型训练进度.md          # LGB、CAB、NN的CV分数、耗时
├── 04-融合结果.md              # Stacking、最终融合的MAE、预测分布
├── 05-参数对照表.md            # 所有超参数与高分方案的对照表
└── 06-最终报告.md              # 成功确认、MAE对标、总结
```

---

## Phase 6: 执行时间表

| 阶段 | 预计耗时 | 关键输出 |
|------|---------|---------|
| Phase 1: 环境 + 代码 | 30分钟 | v2就位 + TF安装✓ |
| Phase 2.1: 预处理 | 30分钟 | combined_preprocessed.csv |
| Phase 2.2: 特征工程 | 45分钟 | train_tree.csv, train_nn.csv |
| Phase 3: 模型训练 | 2.5小时 | LGB/CAB/NN OOF预测 |
| Phase 4: 融合 | 15分钟 | predictions.csv |
| Phase 5: 验证 + 文档 | 30分钟 | 完整报告 |
| **总计** | **~5小时** | **最终提交文件** |

---

## 总结

本计划遵循 `参考高分方案/` 的代码逻辑和参数配置，在当前环境 (Python 3.13.5, 苹果Silicon M4) 上进行适配和执行。所有进度和结果记录在 `v2/recoveryplan/` 中，便于后续验证和微调。

**预期目标**: 404-410 MAE的高分分数

**预期耗时**: 5小时完整执行

**成功标志**: 最终predictions.csv的MAE≤415 (线下估计)
