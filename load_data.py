from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import pandas as pd

class CrowdDataset(Dataset):

    def __init__(self, img_root, gt_dmap_root, gt_downsample=1):
        self.img_root = img_root
        self.gt_dmap_root = gt_dmap_root
        self.gt_downsample = gt_downsample
        self.img_names = [filename for filename in os.listdir(img_root) \
                          if os.path.isfile(os.path.join(img_root, filename))]
        self.n_samples = len(self.img_names)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):  # 实现对象的索引访问，当使用obj[index]来访问某个访问时，实际调用该函数。
        assert  index <= len(self), 'index range error'
        img_name = self.img_names[index]


        # img = plt.imread(os.path.join(self.img_root, img_name))
        #修改 ZT 能快速一些
        img = cv2.imread(os.path.join(self.img_root, img_name),0)
        img = img.astype(np.float32, copy=False)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #x, y = img.shape[0:2]  #自己补充的，用于解决图像过大的问题
        #img = cv2.resize(img,(int(y/2),int(x/2)))

        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate((img, img, img), 2)
        # ZT 修改  1、格式不一样 2  Pytorch通常使用float32作为默认浮点数据类型，转换成float32可以避免数据类型不匹配问题，原来gt_dmap是64位
        gt_dmap = pd.read_csv(os.path.join(self.gt_dmap_root,os.path.splitext(img_name)[0] + '.csv'), sep=',',header=None).values
        gt_dmap = gt_dmap.astype(np.float32, copy=False)
        # gt_dmap = np.load(os.path.join(self.gt_dmap_root, img_name.replace('.jpg', '.npy')))
        #gt_dmap = cv2.resize(gt_dmap,(int(y/2),int(x/2)))
        if self.gt_downsample > 1:
            ds_rows = int(img.shape[0] // self.gt_downsample)    # 高长
            ds_cols = int(img.shape[1] // self.gt_downsample)   # 宽长 横长
            ds_cols = max(ds_cols,1)
            ds_rows = max(ds_rows,1)
            img = cv2.resize(img, (ds_cols * self.gt_downsample, ds_rows * self.gt_downsample))
            img = img.transpose((2, 0, 1))  # convert to order (channel,rows,cols)
            # print(f"Original image size: {img.shape}, Ground truth size:{gt_dmap.shape}")
            gt_dmap = cv2.resize(gt_dmap, (ds_cols, ds_rows))
            # Shanghai 数据集
            gt_dmap = gt_dmap[np.newaxis, :, :] * self.gt_downsample * self.gt_downsample

            img_tensor = torch.tensor(img, dtype=torch.float)
            gt_dmap_tensor = torch.tensor(gt_dmap, dtype=torch.float)

        return img_tensor, gt_dmap_tensor

if __name__=="__main__":
    img_root="data/train_data/images"
    gt_dmap_root="data/train_data/ground_truth"
    dataset=CrowdDataset(img_root,gt_dmap_root,gt_downsample=4)
    for i,(img,gt_dmap) in enumerate(dataset):
        # plt.imshow(img)
        # plt.figure()
        # plt.imshow(gt_dmap)
        # plt.figure()
        # if i>5:
        #     break
        print(img.shape,gt_dmap.shape)