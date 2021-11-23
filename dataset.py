import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
import os
import sys 
import cfg

# 构建数据提取器，利用dataloader
# 利用torchvision中的transforms进行图像预处理
#cfg为config文件，保存几个方便修改的参数

class  SelfCustomDataset(Dataset):
    def __init__(self, root, txt_file, imageset):
        '''
        img_dir: 图片路径：img_dir + img_name.jpg构成图片的完整路径      
        '''
        # 所有图片的绝对路径
        self.root = root
        with open(txt_file, 'r') as f:
            #label_file的格式， （label_file image_label)
            lines = f.readlines()
        self.imgs = list(map(lambda line: line.strip().split(" "), lines))
      # 相关预处理的初始化
      #   self.transforms=transform
        self.img_aug=True
        if imageset == 'train':
            self.transform= transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform= transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.406, 0.456 ], std=[0.229, 0.224, 0.225]),
            ])
        self.input_size = cfg.INPUT_SIZE

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        # print(img_path)
        # img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        img = cv2.resize(cv2.imread(os.path.join(self.root, img_path)), cfg.INPUT_SIZE)[:, :, ::-1]
        if np.random.choice(2):
            img = cv2.rotate(img, cv2.ROTATE_180)
            label = 1-int(label)
        img = Image.fromarray(img)
        if self.img_aug:
            img =self.transform(img)
        else:
            img = np.array(img)
            img = torch.from_numpy(img)

        return img, torch.from_numpy(np.array(int(label)))
 
    def __len__(self):
        return len(self.imgs)


train_datasets = SelfCustomDataset(cfg.BASE, cfg.TRAIN_TXT, imageset='train')
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2)

val_datasets = SelfCustomDataset(cfg.BASE, cfg.VAL_TXT, imageset='test')
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)
