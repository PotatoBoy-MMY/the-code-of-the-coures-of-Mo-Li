import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

#读取热狗数据集
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip','fba480ffa8aa7e0febbb511d181409f899b9baa5')
data_dir = d2l.download_extract('hotdog')

#为文件夹中的数据打标号
train_imgs=torchvision.datasets.ImageFolder(os.path.join(data_dir,'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

# hot_dogs=[train_imgs[i][0]for i in range(8)]
# no_hotdogs=[test_imgs[-i-1][0] for i in range(8)]
#
# d2l.show_images(hot_dogs+no_hotdogs,2,8,scale=1.5)

normalize=torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

# 训练数据要会更加随机，噪声的增加，防止过拟合的出现
train_augs=torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize
])


# 测试集要更加稳定，标准一些
test_augs=torchvision.transforms.Compose([
    torchvision.transforms.Resize([256,256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize
])

pretrained_net=torchvision.models.resnet18(pretrained=True)

finetune_net=torchvision.models.resnet18(pretrained=True)
finetune_net.fc=nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight)


def train_fine_tuning(net,learning_rate,batch_size=128,num_epochs=5,param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_augs), batch_size=batch_size, shuffle=True)
    test_iter=torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_augs), batch_size=batch_size, shuffle=True)
    devices =d2l.try_all_gpus()
    loss=nn.CrossEntropyLoss(reduction='none')
    if param_group:
        param_1x=[param for name,param in net.named_parameters()
                        if name not in ['fc.weight','fc.bias']]
        trainer =torch.optim.SGD(
            [{'params': param_1x},{'params':net.fc.weight,'lr':learning_rate*10}],
            lr=learning_rate,
            weight_decay=0.001
        )
    else:
        trainer=torch.optim.SGD(net.parameters(),lr=learning_rate,weight_decay=0.001)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,devices)


train_fine_tuning(finetune_net, 5e-5)
plt.show()