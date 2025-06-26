import torch
from d2l.tensorflow import Accumulator, accuracy
from torch import nn
from d2l import torch as d2l


#读取数据
batch_size=1
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
#建立模型
net=nn.Sequential(nn.Flatten(),nn.Linear(784,10))
#初始化模型参数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weights)
#损失函数
loss=nn.CrossEntropyLoss(reduction='none')
#优化函数
trainer=torch.optim.SGD(net.parameters(),lr=0.1)
#训练
def train_epoch(net,train_iter,loss,updater):
    if isinstance(net,torch.nn.Module):
        net.train()
    metric=Accumulator(3)
    for x,y in train_iter:
        y_hat=net(x)
        l=loss(y_hat,y)
        updater.zero_grad()
        l.mean().backward()
        updater.step()
    metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    return metric[0]/metric[2],metric[1]/metric[2]


for x,y in train_iter:
    y_hat=net(x)
    l=loss(y_hat,y)
    metric=Accumulator(3)
    metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    print(metric[0]/metric[2],metric[1]/metric[2])


