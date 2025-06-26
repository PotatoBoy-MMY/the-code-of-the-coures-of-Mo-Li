import torch
from torch import nn
from torch.nn.functional import conv2d


def comp_conv2d(conv2d,X):
    X=X.reshape((1,1)+X.shape)
    Y=conv2d(X)
    return Y.reshape(Y.shape[2:])

#卷积核为一个3*3的配合填充1刚好
# conv2d=nn.Conv2d(1,1,kernel_size=3,padding=1)
X=torch.rand(size=(8,8))
# print(comp_conv2d(conv2d,X))

#适配相对应的卷积核来适配相对应的填充大小
# conv2d=nn.Conv2d(1,1,kernel_size=(5,3),padding=(2,1))
# print(comp_conv2d(conv2d,X).shape)

#步幅
conv2d=nn.Conv2d(1,1,kernel_size=3, stride=2, padding=1)
print(comp_conv2d(conv2d,X).shape)


