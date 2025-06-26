import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l

# d2l.set_figsize()
# img=d2l.Image.open('../img/cat1.jpg')
# d2l.plt.show(img)


d2l.set_figsize()
img = d2l.Image.open('C:\\Users\\hahahaha\\Desktop\\ca046a0296305de596cf2b09e3c44aa.jpg')
# d2l.plt.imshow(img)


def apply(img, aug, num_rows=1, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    # return Y
    d2l.show_images(Y, num_rows, num_cols, scale=scale)

# img_list=[]
# 翻转
# img_list.extend(apply(img,torchvision.transforms.RandomHorizontalFlip(),scale=1.5))
# img_list.extend(apply(img,torchvision.transforms.RandomVerticalFlip(),scale=1.5))

# d2l.show_images(img_list,num_rows=2,num_cols=4,scale=1.5)



# # 裁剪
# shape_aug = torchvision.transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))
# # apply(img, shape_aug)
#
#
# apply(img,torchvision.transforms.ColorJitter(brightness=0.5,contrast=0,saturation=0.5,hue=0))
# apply(img,torchvision.transforms.ColorJitter(brightness=0,contrast=0,saturation=0,hue=0.5))
#
# color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
# apply(img, color_aug)
# augs = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
# apply(img, augs)


all_images = torchvision.datasets.CIFAR10(train=True, root="../data",download=True)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)


plt.show()

