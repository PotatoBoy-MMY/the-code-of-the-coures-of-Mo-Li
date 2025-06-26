import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt


#定义一个NiN块
def nin_block(in_channels,out_channels,kernel_size,strides,paddings):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=strides,padding=paddings),
        nn.ReLU(),
        nn.Conv2d(out_channels,out_channels,kernel_size=1),nn.ReLU(),
        nn.Conv2d(out_channels,out_channels,kernel_size=1),nn.ReLU(),
    )

#组建网络
net=nn.Sequential(
    nin_block(1,96,kernel_size=11,strides=4,paddings=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(96,128,kernel_size=3,strides=1,paddings=1),
    nn.MaxPool2d(kernel_size=3, stride=1,padding=1),
    nin_block(128,160,kernel_size=3,strides=1,paddings=1),
    nn.MaxPool2d(kernel_size=3, stride=1,padding=1),
    nin_block(160,192,kernel_size=3,strides=1,paddings=0),
    nn.MaxPool2d(kernel_size=3, stride=1,padding=1),
    nin_block(192,256,kernel_size=3,strides=2,paddings=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(256,384,kernel_size=3,strides=1,paddings=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Dropout2d(p=0.5),
    nin_block(384,10,kernel_size=3,strides=1,paddings=1),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten()
)



lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
plt.show()