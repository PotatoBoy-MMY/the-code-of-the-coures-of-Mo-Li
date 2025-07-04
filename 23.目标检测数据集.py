import os
import pandas as pd
import torch
import torchvision
from d2l import torch as d2l
from matplotlib import pyplot as plt

d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72'
)

def read_data_bananas(is_train=True):

    # 下载并处理数据
    data_dir=d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'label.csv')
    csv_data=pd.read_csv(csv_fname)
    csv_data=csv_data.set_index('img_name')

    # 加载数据及其标签
    images,targets=[],[]
    for img_name,target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'images',f'{img_name}')
        ))
        targets.append(list(target))
    return images,torch.tensor(targets).unsqueeze(1)/256

class BananasDataset(torch.utils.data.Dataset):
    def __init__(self,is_train):
        self.features,self.labels=read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if is_train else f' validation examples'))

    def __getitem__(self,idx):
        return self.features[idx].float(),self.labels[idx]

    def __len__(self):
        return len(self.labels)

def load_data_bananas(batch_size):
    train_iter=torch.utils.data.DataLoader(BananasDataset(is_train=True),batch_size=batch_size,shuffle=True)
    val_iter=torch.utils.data.DataLoader(BananasDataset(is_train=False),batch_size=batch_size,shuffle=False)
    return train_iter,val_iter

# 批量大小及
batch_size,edge_size=32,256
train_iter,_=load_data_bananas(batch_size)
batch=next(iter(train_iter))

imgs=(batch[0][0:10].permute(0,2,3,1))/255
axes =d2l.show_images(imgs,2,5,scale=2)
for ax,label in zip(axes,batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])

plt.show()

