import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import matplotlib.pyplot as plt

batch_size,num_steps=32,25
train_iter,vocab=d2l.load_data_time_machine(batch_size=batch_size,num_steps=num_steps)


#初始化权重
def get_params(vocab_size,num_hiddens,device):
    num_inputs = num_outputs = vocab_size #来自同一个词表，大小相同

    def normal(shape):
        return torch.randn(size=shape,device=device)*0.01       #用于后续整合出初始化权重方法

    #用于更新下一个隐变量
    W_xh=normal((num_inputs,num_hiddens))
    W_hh=normal((num_hiddens,num_hiddens))
    b_h=torch.zeros(num_hiddens,device=device)

    #用于输出
    W_hq=normal((num_hiddens,num_outputs))
    b_q=torch.zeros(num_outputs,device=device)

    params=[W_xh, W_hh, b_h, W_hq, b_q]

    for p in params:
        p.requires_grad_(True)

    return params

#用于初始化隐状态
def init_rnn_state(batch_size,num_hiddens,device):
    return (torch.zeros((batch_size,num_hiddens),device=device),)


#定义RNN更新隐状态和输出的方法
def rnn(inputs,state,params):
    W_xh, W_hh, b_h, W_hq, b_q=params
    H,=state
    outputs=[]
    for X in inputs:
        H=torch.tanh(torch.mm(X,W_xh)+torch.mm(H,W_hh)+b_h)
        Y=torch.mm(H,W_hq)+b_q
        outputs.append(Y)
    return torch.cat(outputs,dim=0),(H,)


class RNNModelScratch:
    def __init__(self,vocab_size,num_hiddens,device,get_params,init_state,forward_fn):
        self.vocab_size=vocab_size
        self.num_hiddens=num_hiddens
        self.params=get_params(vocab_size,num_hiddens,device)
        self.init_state=init_state
        self.forward_fn=forward_fn

    def __call__(self,X,state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)



#预测函数
def predict_ch8(prefix,num_preds,net,vocab,device):
    #RNN中是一直对一个隐变量进行更新
    state=net.begin_state(batch_size=1,device=device)
    outputs=[vocab[prefix[0]]]

    #从函数的最末尾取出字符，作为新一轮的输入
    get_input=lambda:torch.tensor([outputs[-1]],device=device).reshape((1,1))
    for y in prefix[1:]:
        _,state=net(get_input(),state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y,state=net(get_input(),state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i]for i in outputs])

predict_ch8('time traveller',10,net,vocab,d2l.try_gpu())