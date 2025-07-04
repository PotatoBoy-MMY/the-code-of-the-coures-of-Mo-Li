import random
import torch
from d2l import torch as d2l
from torchgen.api import autograd


def synthetic_data(w,b,num_examples):
    X=torch.normal(0,1,(num_examples,len(w)))
    y=torch.matmul(X,w)+b
    y+=torch.normal(0,0.01,y.shape)
    return X,y.reshape((-1,1))

true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=synthetic_data(true_w, true_b, 1000)


def data_iter(batch_size,features, labels):
    num_examples=len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices]

"""模型的初始化"""
w=torch.normal(0,0.1,size=(2,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)
batch_size = 10
def linreg(X,w,b):
    return torch.matmul(X,w) + b

'''损失函数'''
def squared_loss(y_hat, y):
    return (y_hat-y.reshape(y.shape))**2/2

'''优化函数'''
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad/batch_size
            param.grad.zero_()


lr=0.03
num_epochs=3
net=linreg
loss=squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l=loss(net(X,w,b),y)
        l.sum().backward()
        sgd([w,b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
