import torch
from d2l import torch as d2l

#多输入通道的互相关函数
def corr2d_multi_in(X,K):
    return sum(d2l.corr2d(x,k) for x,k in zip(X,K))
# def corr2d_multi_in(X,K):
#     return [d2l.corr2d(x,k) for x,k in zip(X,K)]
X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
print(corr2d_multi_in(X, K))


#多通道输出
#通俗理解有六个卷积核，每两个为一组将原本的输入数据中的两个书法如数据压成一个输出，所以产生三个通道的输出
#但正式的理解应该是K其实是一个四维的张量，第一维对应每个输出通道的卷积核组，第二维对应每个卷积核组里面有多少个卷积核用于对应输入通道数，第三第四维度为卷积核的宽和高
def corr2d_multi_in_out(X,K):
    return torch.stack([corr2d_multi_in(X,k) for k in K],0)#取出不同维度的卷积核来与数据卷积然后堆叠在第0维上

K = torch.stack((K, K + 1, K + 2), 0)
# print(corr2d_multi_in_out(X, K))

#使用1*1的卷积核进行特征融合
def corr2d_multi_in_out_1x1(X,K):
    c_i,h,w=X.shape
    c_o=K.shape[0]
    X=X.reshape((c_i,h*w))
    K=K.reshape((c_o,c_i))
    Y=torch.matmul(K,X)
    return Y.reshape((c_o,h,w))

X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
print(Y1,Y2)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6
