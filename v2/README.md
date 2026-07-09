# v2: 高分方案

**项目名称**: v2
**版本**: v0.1 (执行前准备完成)
**目标**: 高分方案，达成404-410 MAE目标
**环境**: Python 3.13.5, Apple M4 Silicon, 24GB内存, 无GPU
**执行状态**: ✅ **完全就位，可立即开始执行**

---

## 📋 项目概况

本项目是 高分方案的和适配项目，使用统一的数据源 `/Case-二手车价格预测-data/`，在当前环境 (Python 3.13.5, Apple M4) 上执行。

### 核心目标
- **分数目标**: 404-410 MAE (线下估计)
- **方案**: 三层集成 (LGB 10-Fold + CAB 10-Fold + NN 5-Fold → Stacking → 最终融合)
- **特征**: 双轨路径 (树特征83个 + NN特征29个)
- **融合方式**: Bayesian Ridge Stacking + 0.5加权融合

---

## 📁 项目结构

```
v2/
├── plans/                                  # 计划与指南
│   └── 高分方案计划.md            # 完整执行计划
│
├── code/                                   # 源代码（参考自高分方案）
│   ├── feature/                            # 特征工程
│   │   ├── generation.py                   # 通用预处理 (→ combined_preprocessed.csv)
│   │   ├── Tree_generation.py              # 树特征工程 (→ train_tree/test_tree.csv, 83列)
│   │   └── NN_generation.py                # NN特征工程 (→ train_nn/test_nn.csv, 29列)
│   ├── model/                              # 模型训练与融合
│   │   ├── lgb_model.py                    # LightGBM 10-Fold CV (→ lgb_train/test.csv)
│   │   ├── cab_model.py                    # CatBoost 10-Fold CV (→ cab_train/test.csv)
│   │   ├── nn_model.py                     # Neural Network 5-Fold CV (→ nn_train/test.csv)
│   │   ├── model.py                        # 完整管道 (可选入口)
│   │   ├── stack+mix.py                    # Stacking + 最终融合 (→ predictions.csv)
│   │   └── requirements.txt                # 依赖清单
│
├── data/                                   # 数据目录
│   ├── raw/                                # 原始数据 (符号链接到 Case-二手车价格预测-data/)
│   │   ├── used_car_train_20200313.csv    # 训练数据 (150K行)
│   │   ├── used_car_testB_20200421.csv    # 测试数据 (50K行)
│   │   └── used_car_sample_submit.csv     # 提交样本
│   └── processed/                          # 处理后的特征（存储位置）
│
├── results/                                # 输出目录
│   ├── submissions/                        # 最终提交
│   │   └── predictions.csv                 # 最终预测 (50K行×2列)
│   └── metrics/                            # 评估数据
│
├── user_data/                              # 模型中间输出（自动生成）
│   ├── combined_preprocessed.csv           # Phase 1输出
│   ├── train_tree.csv, test_tree.csv       # Phase 2A输出
│   ├── train_nn.csv, test_nn.csv           # Phase 2B输出
│   ├── lgb_train.csv, lgb_test.csv         # Phase 3.1输出
│   ├── cab_train.csv, cab_test.csv         # Phase 3.2输出
│   ├── nn_train.csv, nn_test.csv           # Phase 3.3输出
│   └── [最后由stack+mix生成最终输出]
│
├── recoveryplan/                         # 执行日志与文档
│   ├── 00-执行前检查清单.md                # 执行前准备清单
│   ├── 01-环境准备日志.md                  # 环境配置与TensorFlow问题处理
│   ├── 02-特征工程进度.md                  # [待执行后生成]
│   ├── 03-模型训练进度.md                  # [待执行后生成]
│   ├── 04-融合结果.md                      # [待执行后生成]
│   ├── 05-参数对照表.md                    # [待执行后生成]
│   └── 06-最终报告.md                      # [待执行后生成]
│
└── README.md                               # 本文件
```

---

## 🚀 快速启动

### 环境检查 (2分钟)

```bash
# 1. 验证Python版本
python3 --version                          # 应为 3.13.5

# 2. 验证关键依赖
python3 -c "import pandas, numpy, lightgbm, catboost, torch; print('✅ All dependencies OK')"

# 3. 验证数据文件
ls -lh /Users/macpro/Downloads/ai/day18-分析式AI基础/v2/data/raw/
# 应显示 train_20200313.csv (52M), testB_20200421.csv (17M), sample_submit.csv (439K)
```

### 执行Phase 1: 数据预处理 (30分钟)

```bash
cd /Users/macpro/Downloads/ai/day18-分析式AI基础/v2/code/feature
python generation.py

# 预期输出: ../user_data/combined_preprocessed.csv
#   - Shape: (200001, 29-31) [150K train + 50K test]
#   - 无NaN值
```

### 执行Phase 2: 特征工程 (18分钟)

```bash
# Phase 2A: 树特征 (15分钟)
python Tree_generation.py
# 输出: train_tree.csv (150K×83), test_tree.csv (50K×83)

# Phase 2B: NN特征 (3分钟)
python NN_generation.py
# 输出: train_nn.csv (150K×29), test_nn.csv (50K×29), 值在[0,1]
```

### 执行Phase 3: 模型训练 (90分钟，可并行)

```bash
cd ../model

# 方式1: 串行执行
python lgb_model.py      # 40分钟 → lgb_train.csv, lgb_test.csv (预期MAE: 420-430)
python cab_model.py      # 50分钟 → cab_train.csv, cab_test.csv (预期MAE: 420-430)
python nn_model.py       # 45分钟 → nn_train.csv, nn_test.csv (预期MAE: 420-430)

# 方式2: 并行执行（后台）
python lgb_model.py &
python cab_model.py &
python nn_model.py &
wait
```

### 执行Phase 4: 融合与提交 (15分钟)

```bash
python stack+mix.py
# 输出: ../results/submissions/predictions.csv
#   - Shape: (50000, 2) [SaleID, price]
#   - 预期融合MAE: 404-415
```

### 完整执行时间线

| Phase | 任务 | 预计时间 | 总耗时 |
|-------|------|---------|--------|
| 1 | 数据预处理 | 30分钟 | 30分钟 |
| 2A | 树特征工程 | 15分钟 | 45分钟 |
| 2B | NN特征工程 | 3分钟 | 48分钟 |
| 3 | 模型训练 | 90分钟 | 138分钟 |
| 4 | 融合提交 | 15分钟 | 153分钟 |
| **总计** | **完整执行** | **~2.5-3小时** | |

---

## ✅ 执行前检查清单

在开始执行前，请按照 `recoveryplan/00-执行前检查清单.md` 的清单逐项确认：

- [x] Python 3.13.5 已安装并可用
- [x] 所有必需依赖已安装: pandas, numpy, lightgbm, catboost, torch
- [x] TensorFlow 2.20.0 已安装（备选）
- [x] 代码文件已迁移到 v2/code/
- [x] 数据文件已链接到 v2/data/raw/
- [x] 文件引用已更新: testA→testB, text_tree→test_tree
- [x] 路径逻辑已验证
- [x] user_data/ 目录已创建

**状态**: ✅ **完全就位，可立即执行**

---

## 🔍 关键文档

### 立即查看
1. **执行前检查清单**: `recoveryplan/00-执行前检查清单.md`
   - 完整的系统检查项
   - 执行前验证命令
   - 已知问题与应对

2. **环境准备日志**: `recoveryplan/01-环境准备日志.md`
   - 环境配置详情
   - TensorFlow安装过程
   - 分阶段执行计划与预期结果
   - 验证命令与风险处理

3. **原始计划**: `plans/高分方案计划.md`
   - 完整的6 Phase执行计划
   - 详细的参数对照表
   - 故障排查策略

---

## 🔨 执行过程中的操作

### 记录执行进度
执行每个Phase后，更新对应的进度文档:
- Phase 1 完成 → 更新 `recoveryplan/02-特征工程进度.md`
- Phase 2 完成 → 更新 `recoveryplan/02-特征工程进度.md`
- Phase 3 完成 → 更新 `recoveryplan/03-模型训练进度.md`
- Phase 4 完成 → 更新 `recoveryplan/04-融合结果.md`

### 常见问题排查
如遇到错误，参考 `recoveryplan/01-环境准备日志.md` 的"风险与应对"部分。

### 最终报告
全部执行完成后，生成 `recoveryplan/06-最终报告.md` 确认:
- 完整执行时间
- 各阶段实际MAE vs 预期
- 是否达成404-410目标
- 问题总结与后续建议

---

## 📊 性能目标

### 单模型预期
- **LightGBM**: 420-430 MAE (10-Fold CV)
- **CatBoost**: 420-430 MAE (10-Fold CV)
- **Neural Network**: 420-430 MAE (5-Fold CV)

### 融合预期
- **Stacking (BR)**: 410-420 MAE
- **最终融合 (0.5Stack + 0.5NN)**: 404-415 MAE (线下估计)

### 上线评估
- 在竞赛平台上评估实际MAE

---

## ⚠️ 特殊说明

### TensorFlow 与 PyTorch
- TensorFlow 2.20.0 已安装但存在arm64导入问题
- nn_model.py 可改用 PyTorch (2.9.1, Metal支持) 替代
- 两者性能预期接近，PyTorch在此环境可能更稳定

### 路径处理
- 所有代码使用相对路径，自动定位 user_data
- 可从任意子目录执行，路径计算自动调整
- 若遇路径问题，确保从 v2/ 或其直接子目录执行

### 数据一致性
- 使用统一的 `/Case-二手车价格预测-data/` 数据源
- 避免与 mcode-v13 混用（完全独立项目）
- 防泄漏: source 列标记train/test，统计仅用train数据

---

## 📝 文档更新历程

| 日期 | 内容 | 完成度 |
|-----|------|--------|
| 2025-12-04 | 环境准备 + 代码迁移 + 路径调整 | ✅ 100% |
| 2025-12-04 | 创建执行前清单与日志 | ✅ 100% |
| [执行中] | 每Phase完成后更新进度 | 待执行 |
| [最后] | 生成最终报告 | 待执行 |

---

## 🎯 成功标志

✅ **项目完全就位**:
- 环境配置完成
- 代码修正完成
- 数据链接完成
- 文档准备完成

✅ **可立即执行**:
- 运行 `cd code/feature && python generation.py` 开始Phase 1
- 预期2.5-3小时内完成全部执行
- 预期达成404-410 MAE目标（线下估计）

---

## 🚀 立即开始

```bash
# 进入项目目录
cd /Users/macpro/Downloads/ai/day18-分析式AI基础/v2

# 查看执行前清单
cat recoveryplan/00-执行前检查清单.md

# 开始Phase 1
cd code/feature
python generation.py
```

---

**项目准备完毕，等待执行** 🚀

