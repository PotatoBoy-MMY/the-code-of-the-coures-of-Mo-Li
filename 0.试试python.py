
"""继承变量访问以及双下划线用法"""
# class Myclass:
#     def __init__(self):
#         self.__mmy=10
#
# class SonClass(Myclass):
#     def __init__(self):
#         super().__init__()
#
#     def printClass(self):
#         print(self._Myclass__mmy)
#
# obj=Myclass()
# obj2=SonClass()
#
# print(obj._Myclass__mmy)
# # print(obj2._SonClass__mmy)
# obj2.printClass()

# import os
# import pandas as pd
# import torch
# import torchvision
# from d2l import torch as d2l
#
# #@save
# d2l.DATA_HUB['banana-detection'] = (
#     d2l.DATA_URL + 'banana-detection.zip',
#     '5de26c8fce5ccdea9f91267273464dc968d20d72')
#
# #@save
# def read_data_bananas(is_train=True):
#     """读取香蕉检测数据集中的图像和标签"""
#     data_dir = d2l.download_extract('banana-detection')
#     csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
#                              else 'bananas_val', 'label.csv')
#     csv_data = pd.read_csv(csv_fname)
#     csv_data = csv_data.set_index('img_name')
#     images, targets = [], []
#     for img_name, target in csv_data.iterrows():
#         images.append(torchvision.io.read_image(
#             os.path.join(data_dir, 'bananas_train' if is_train else
#                          'bananas_val', 'images', f'{img_name}')))
#         # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
#         # 其中所有图像都具有相同的香蕉类（索引为0）
#         targets.append(list(target))
#     return images, torch.tensor(targets).unsqueeze(1) / 256
#
# #@save
# class BananasDataset(torch.utils.data.Dataset):
#     """一个用于加载香蕉检测数据集的自定义数据集"""
#     def __init__(self, is_train):
#         self.features, self.labels = read_data_bananas(is_train)
#         print('read ' + str(len(self.features)) + (f' training examples' if
#               is_train else f' validation examples'))
#     def __getitem__(self, idx):
#         return (self.features[idx].float(), self.labels[idx])
#
#     def __len__(self):
#         return len(self.features)
#
#
# #@save
# def load_data_bananas(batch_size):
#     """加载香蕉检测数据集"""
#     train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
#                                              batch_size, shuffle=True)
#     val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
#                                            batch_size)
#     return train_iter, val_iter
#
#
# batch_size, edge_size = 32, 256
# train_iter, _ = load_data_bananas(batch_size)
# batch = next(iter(train_iter))
#
# # setattr()
#
#
# import torch
# hello=[1,2,3]
# H,=hello
# print(H)
# H=hello
# print(H)




import torch
import torch.nn as nn
import torch.nn.functional as F

# 一条样本：2个输入特征
x = torch.tensor([[0.5, -1.2]], requires_grad=True)   # shape: (1, 2)
y_true = torch.tensor([1])  # 标签类别为1

# 定义模型参数（手动写，不用nn.Module）
W1 = torch.randn(3, 2, requires_grad=True)  # shape: (3, 2)
b1 = torch.randn(3, requires_grad=True)     # shape: (3,)
W2 = torch.randn(2, 3, requires_grad=True)  # shape: (2, 3)
b2 = torch.randn(2, requires_grad=True)     # shape: (2,)

# === Forward 前向传播 ===

# 第一层线性变换：z1 = x @ W1.T + b1
z1 = x @ W1.T + b1       # shape: (1, 3)
a1 = F.relu(z1)          # shape: (1, 3)

# 第二层线性变换：z2 = a1 @ W2.T + b2
z2 = a1 @ W2.T + b2      # shape: (1, 2)

# Softmax + CrossEntropyLoss（自动融合）
loss = F.cross_entropy(z2, y_true)  # 自动计算softmax + log + loss

print(f"Loss: {loss.item()}")

# === Backward 反向传播 ===
loss.backward()

# === 打印梯度 ===
print("∂Loss/∂W2:", W2.grad)   # shape: (2, 3)
print("∂Loss/∂W1:", W1.grad)   # shape: (3, 2)
print("∂Loss/∂x :", x.grad)    # shape: (1, 2)
