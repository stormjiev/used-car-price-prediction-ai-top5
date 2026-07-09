## 树模型Stacking（不依赖NN，可并行执行）
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
tree_data_path = path+'/user_data/'

print("=" * 60)
print("树模型Stacking（LGB + CAB → Bayesian Ridge）")
print("=" * 60)

# 导入树模型lgb预测数据
predictions_lgb = np.array(pd.read_csv(tree_data_path+'lgb_test.csv')['price'])
oof_lgb = np.array(pd.read_csv(tree_data_path+'lgb_train.csv')['price'])
print(f"LightGBM: train OOF shape={oof_lgb.shape}, test shape={predictions_lgb.shape}")

# 导入树模型cab预测数据
predictions_cb = np.array(pd.read_csv(tree_data_path+'cab_test.csv')['price'])
oof_cb = np.array(pd.read_csv(tree_data_path+'cab_train.csv')['price'])
print(f"CatBoost: train OOF shape={oof_cb.shape}, test shape={predictions_cb.shape}")

# 读取price，对验证集进行评估
Train_data = pd.read_csv(tree_data_path+'train_tree.csv', sep=' ')
TestA_data = pd.read_csv(tree_data_path+'test_tree.csv', sep=' ')
Y_data = Train_data['price']
print(f"Target: train shape={Y_data.shape}")

# 单模型评估（log空间 → 原始空间MAE）
lgb_mae = mean_absolute_error(np.expm1(oof_lgb), np.expm1(Y_data))
cab_mae = mean_absolute_error(np.expm1(oof_cb), np.expm1(Y_data))
print(f"\n单模型MAE:")
print(f"  LightGBM: {lgb_mae:.2f}")
print(f"  CatBoost: {cab_mae:.2f}")

# 构建stacking特征
train_stack = np.vstack([oof_lgb, oof_cb]).transpose()
test_stack = np.vstack([predictions_lgb, predictions_cb]).transpose()
print(f"\nStacking特征: train={train_stack.shape}, test={test_stack.shape}")

# 二层贝叶斯回归stack（RepeatedKFold: 10折×2次）
folds_stack = RepeatedKFold(n_splits=10, n_repeats=2, random_state=2018)
tree_stack_oof = np.zeros(train_stack.shape[0])
tree_stack_pred = np.zeros(test_stack.shape[0])

print("\n开始Bayesian Ridge Stacking (10折×2次)...")
for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, Y_data)):
    trn_data, trn_y = train_stack[trn_idx], Y_data.iloc[trn_idx]
    val_data, val_y = train_stack[val_idx], Y_data.iloc[val_idx]

    Bayes = linear_model.BayesianRidge()
    Bayes.fit(trn_data, trn_y)
    tree_stack_oof[val_idx] = Bayes.predict(val_data)
    tree_stack_pred += Bayes.predict(test_stack) / 20

    if (fold_ + 1) % 5 == 0:
        print(f"  Fold {fold_+1}/20 completed")

# 转换回原始价格空间
tree_predictions = np.expm1(tree_stack_pred)
tree_stack_oof_exp = np.expm1(tree_stack_oof)
tree_stack_mae = mean_absolute_error(tree_stack_oof_exp, np.expm1(Y_data))

print(f"\n树模型Stacking MAE: {tree_stack_mae:.2f}")

# 简单平均blend对比
blend_oof = 0.5 * oof_lgb + 0.5 * oof_cb
blend_pred = 0.5 * predictions_lgb + 0.5 * predictions_cb
blend_mae = mean_absolute_error(np.expm1(blend_oof), np.expm1(Y_data))
print(f"简单平均Blend MAE: {blend_mae:.2f}")

# 保存树模型Stacking结果
output_path = tree_data_path

# 保存Stacking OOF预测（log空间）
sub_train = pd.DataFrame()
sub_train['SaleID'] = Train_data['SaleID']
sub_train['price'] = tree_stack_oof
sub_train.to_csv(output_path+'tree_stack_train.csv', index=False)
print(f"\n保存: tree_stack_train.csv (OOF, log空间)")

# 保存Stacking测试集预测（原始价格空间，可直接提交）
sub_test = pd.DataFrame()
sub_test['SaleID'] = TestA_data['SaleID']
tree_predictions[tree_predictions < 0] = 0
sub_test['price'] = tree_predictions
sub_test.to_csv(output_path+'tree_stack_test.csv', index=False)
print(f"保存: tree_stack_test.csv (原始价格，可提交)")

# 也生成一个可直接提交的文件
os.makedirs(path + '/prediction_result/', exist_ok=True)
sub_test.to_csv(path + '/prediction_result/tree_stack_submit.csv', index=False)
print(f"保存: prediction_result/tree_stack_submit.csv (可直接提交)")

print("\n" + "=" * 60)
print("树模型Stacking完成!")
print(f"  LightGBM单模型: {lgb_mae:.2f} MAE")
print(f"  CatBoost单模型: {cab_mae:.2f} MAE")
print(f"  简单平均Blend:  {blend_mae:.2f} MAE")
print(f"  Bayesian Stack: {tree_stack_mae:.2f} MAE")
print("=" * 60)
