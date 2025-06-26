from functools import lru_cache
import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l

net=nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5,padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(2,stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(2,stride=2),
    nn.Flatten(),
    nn.Linear(16*5*5, 120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
    )

# 实验各层的输出
# X=torch.rand(size=(1,1,28,28),dtype=torch.float32)
# for layer in net:
#     X=layer(X)
#     print(layer.__class__.__name__,'output shape:\t',X.shape)

batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size=batch_size)


def evaluate_accuracy_gpu(net,data_iter,device=None):
    if isinstance(net,nn.Module):
        net.eval()
        if not device:
            device = next(net.parameters()).device
    metric=d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X,list):
                X=[x.to(device) for x in X]
            else:
                X=X.to(device)
            y=y.to(device)
            metric.add(d2l.accuracy(net(X),y),y.numel())
        return metric[0]/metric[1]


#训练函数
def train_ch6(net,train_iter,test_iter,num_epochs,lr,device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight) #使用均匀分布的初始化权重，适合Sigmoid和Tanh激活函数
    net.apply(init_weights)
    print('training on',device)
    net.to(device)

    #初始化优化器
    optimizer = torch.optim.SGD(net.parameters(),lr=lr)

    #初始化损失函数
    loss=nn.CrossEntropyLoss()

    #绘图函数
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],legend=['train loss', 'train acc', 'test acc'])
    timer,num_batches=d2l.Timer(),len(train_iter)

    #训练部分
    for epoch in range(num_epochs):

        #初始化累加器
        metric=d2l.Accumulator(3)
        #开始训练
        net.train()
        for i,(X,y) in enumerate(train_iter):
            metric=d2l.Accumulator(3)

            #训练的核心部分
            net.train()
            optimizer.zero_grad()
            X,y=X.to(device),y.to(device)
            y_hat=net(X)
            l=loss(y_hat,y)
            l.backward()
            optimizer.step()

            #计算损失率等
            with torch.no_grad():
                metric.add(l*X.shape[0],d2l.accuracy(y_hat,y),X.shape[0])
            timer.stop()

            #计算数据集上的损失和准确率
            train_l=metric[0]/metric[2]
            train_acc=metric[1]/metric[2]

            #绘制图片
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,(train_l, train_acc, None))

        test_acc=evaluate_accuracy_gpu(net,test_iter)
        animator.add(epoch + 1,(None, None, test_acc))
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, 'f'test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec 'f'on {str(device)}')



lr,num_epochs=0.5,40
train_ch6(net,train_iter,test_iter,num_epochs,lr,d2l.try_gpu())
plt.show()