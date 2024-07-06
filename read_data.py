from torch.utils.data import Dataset
from PIL import Image
import os
import time


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir  # 根路径
        self.label_dir = label_dir  # 标签路径
        self.path = os.path.join(self.root_dir, self.label_dir)  # 路径连接
        self.img_path = os.listdir(self.path)  # 影像路径，将self.path转换成为列表

    def __getitem__(self, idx):
        img_name = self.img_path[idx]  # 图片名称
        img_item_path = os.path.join(self.path, img_name)  # 路径加上名字为图片的路径
        img = Image.open(img_item_path)  # 读取图片
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)  # 返回列表长度，也就是数据集合的长度


root_dir = 'dataset/train'
ants_label_dir = 'ants'
bees_label_dir = 'bees'
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)
train_dataset = ants_dataset + bees_dataset  # 数据集合直接连接，尾首相接

print(len(ants_dataset), '+', len(bees_dataset), '==', len(train_dataset))
# 数据集的长度
