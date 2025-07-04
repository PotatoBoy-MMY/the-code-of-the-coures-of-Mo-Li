import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

#实验数据的生成
true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=d2l.synthetic_data(true_w, true_b,1000)

#批量数据
def load_array(data_arrays,batch_size,is_train=True):
    dataset=data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train) #这一步解包为一个分X，y的小批量
batch_size=10
data_iter=load_array((features,labels),batch_size)

#定义模型
net=nn.Sequential(nn.Linear(2,1))

#模型参数设定
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

#损失函数
loss=nn.MSELoss()

#优化函数
trainer=torch.optim.SGD(net.parameters(),lr=0.03)


num_epochs=3
for epoch in range(num_epochs):
    for X,y in data_iter:
        l=loss(net(X),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')



