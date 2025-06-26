import torch
from matplotlib.pyplot import ylabel
from torch import nn
from d2l import torch as d2l

#2维卷积函数
def corr2d(x,k):
    h,w=k.shape
    y=torch.zeros(x.shape[0]-h+1,x.shape[1]-w+1)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i,j]=((x[i:i+h,j:j+w])*k).sum()
    return y


#定义一个卷积层
class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

        def forward(self,x):
            return corr2d(x,self.weight)+self.bias

# 实验卷积函数分辨边缘
X = torch.ones((6, 8))
X[:, 2:6] = 0
# print(X)
K=torch.tensor([[1.0,-1.0]])
Y=corr2d(X,K)
# print(Y)

conv2d=nn.Conv2d(1,1,kernel_size=(1,2),bias=False)
X=X.reshape((1,1,6,8))
Y=Y.reshape((1,1,6,7))
lr=3e-2

for i in range(10):
    Y_hat=conv2d(X)
    l=(Y_hat-Y)**2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:]-=lr*conv2d.weight.grad
    if(i+1)%2==0:
        print(f'epoch{i+1}，loss{l.sum():.3f}')

