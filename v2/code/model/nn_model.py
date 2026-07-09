## 基础工具
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import os

## 读取神经网络模型数据
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
tree_data_path = path+'/user_data/'
Train_NN_data = pd.read_csv(tree_data_path+'train_nn.csv', sep=' ')
Test_NN_data = pd.read_csv(tree_data_path+'test_nn.csv', sep=' ')

numerical_cols = Train_NN_data.columns
feature_cols = [col for col in numerical_cols if col not in ['price','SaleID']]

## 提前特征列，标签列构造训练样本和测试样本
X_data = Train_NN_data[feature_cols]
X_test = Test_NN_data[feature_cols]

x = np.array(X_data, dtype=np.float32)
y = np.array(Train_NN_data['price'], dtype=np.float32)
x_test = np.array(X_test, dtype=np.float32)

## 定义神经网络模型
class NeuralNetModel(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# 设置设备
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# KFold交叉验证
kfolder = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_nn = np.zeros(len(x))
predictions_nn = np.zeros(len(x_test))
predictions_train_nn = np.zeros(len(x))

fold_scores = []
kfold = kfolder.split(x, y)

for fold_idx, (train_index, vali_index) in enumerate(kfold):
    print(f"\nFold {fold_idx + 1}/5")

    k_x_train = x[train_index]
    k_y_train = y[train_index]
    k_x_vali = x[vali_index]
    k_y_vali = y[vali_index]

    # 创建数据加载器
    train_dataset = TensorDataset(torch.from_numpy(k_x_train).to(device),
                                   torch.from_numpy(k_y_train).reshape(-1, 1).to(device))
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    # 初始化模型
    model = NeuralNetModel(x.shape[1]).to(device)
    criterion = nn.L1Loss()  # MAE loss
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.02)

    # 学习率调度
    def adjust_lr(epoch):
        if epoch == 1400:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            print(f"LR adjusted to {param_group['lr']}")
        elif epoch == 1700:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            print(f"LR adjusted to {param_group['lr']}")
        elif epoch == 1900:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            print(f"LR adjusted to {param_group['lr']}")

    # 转换验证数据到 GPU/MPS
    k_x_vali_tensor = torch.from_numpy(k_x_vali).to(device)
    k_y_vali_tensor = torch.from_numpy(k_y_vali).reshape(-1, 1).to(device)

    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 300

    for epoch in range(2000):
        adjust_lr(epoch)

        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(k_x_vali_tensor)
            val_loss = criterion(val_pred, k_y_vali_tensor).item()

        if (epoch + 1) % 300 == 0:
            print(f"Epoch {epoch + 1}: train_loss={train_loss/len(train_loader):.6f}, val_loss={val_loss:.6f}")

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # 生成预测
    model.eval()
    with torch.no_grad():
        k_x_vali_tensor = torch.from_numpy(k_x_vali).to(device)
        vali_pred = model(k_x_vali_tensor).cpu().numpy().reshape(-1)
        oof_nn[vali_index] = vali_pred

        x_test_tensor = torch.from_numpy(x_test).to(device)
        test_pred = model(x_test_tensor).cpu().numpy().reshape(-1)
        predictions_nn += test_pred / kfolder.n_splits

        x_tensor = torch.from_numpy(x).to(device)
        train_pred = model(x_tensor).cpu().numpy().reshape(-1)
        predictions_train_nn += train_pred / kfolder.n_splits

    fold_mae = mean_absolute_error(vali_pred, k_y_vali)
    fold_scores.append(fold_mae)
    print(f"Fold {fold_idx + 1} MAE: {fold_mae:.6f}")

print(f"\nNN overall score: {mean_absolute_error(oof_nn, y):.8f}")
print(f"Fold scores: {fold_scores}")

output_path = path + '/user_data/'

# 测试集输出
predictions = predictions_nn
predictions[predictions < 0] = 0
sub = pd.DataFrame()
sub['SaleID'] = Test_NN_data.SaleID
sub['price'] = predictions
sub.to_csv(output_path+'nn_test.csv', index=False)

# 验证集输出
oof_nn[oof_nn < 0] = 0
sub = pd.DataFrame()
sub['SaleID'] = Train_NN_data.SaleID
sub['price'] = oof_nn
sub.to_csv(output_path+'nn_train.csv', index=False)
