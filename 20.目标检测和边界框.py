import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt


d2l.set_figsize()
img = d2l.Image.open('fries_bird.jpg')
d2l.plt.imshow(img)

# 由左上和右下
def box_corner_to_center(boxes):
    x1,y1,x2,y2=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    cx=(x1+x2)/2
    cy=(y1+y2)/2
    w=x2-x1
    h=y2-y1
    boxes=torch.stack((cx,cy,w,h),axis=-1)
    return boxes

def box_center_to_corner(boxes):
    cx,cy,w,h=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    x1=cx-w/2
    y1=cy-h/2
    x2=cx+w/2
    y2=cy+h/2
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

bird_boxes=[250,850,1100,1200]
boxes=torch.tensor([[900,300,1250,1250]])


def bbox_to_rect(bbox,color):
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],fill=False, edgecolor=color, linewidth=2)

fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(bird_boxes, 'blue'))


plt.show()


