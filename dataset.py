# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import random
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.transforms.functional as F
import glob


class CrowdDataset(torch.utils.data.Dataset):
    '''
    CrowdDataset
    '''

    def __init__(self, root, phase, main_transform=None, img_transform=None):
        '''
        root: the root path of dataset.
        phase: train or test.
        main_transform: transforms on both image and density map.
        img_transform: transforms on image.
        dmap_transform: transforms on densitymap.
        '''
        self.classes = os.listdir(root)
        self.classes = [i for i in self.classes if not i.startswith('.')]
        self.imgpath = [f"{root}/{className}/{'images'}" for className in self.classes]
        self.data_files = [glob.glob(f"{x}/*") for x in self.imgpath]

        files = []
        for i, className in enumerate(self.classes):
            for fileName in self.data_files[i]:
                files.append([i, className, fileName])
        self.data_files = files
        files = None
        self.main_transform = main_transform
        self.img_transform = img_transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        index = index % len(self.data_files)
        fname = self.data_files[index]
        img = self.read_image_and_dmap(fname)
        if self.main_transform is not None:
            img = self.main_transform((img))
        if self.img_transform is not None:
            img = self.img_transform(img)

        return {'image': img}

    def __getitem__(self, idx):

        fileName = self.data_files[idx][2]
        classCategory = self.data_files[idx][0]
        img = self.read_image_and_dmap(fileName)
        if self.main_transform is not None:
            img = self.main_transform((img))
        if self.img_transform is not None:
            img = self.img_transform(img)
            
        return {'image': img, 'class': classCategory}

    def read_image_and_dmap(self, fileName):
        img = Image.open(os.path.join(fileName))
        if img.mode == 'L':
            print('There is a grayscale image.')
            img = img.convert('RGB')

        return img

class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255                       # 避免有值超过255而反转
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

class AddSaltPepperNoise(object):

    def __init__(self, density=0):
        self.density = density

    def __call__(self, img):

        img = np.array(img)                                                             # 图片转numpy
        h, w, c = img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd/2.0, Nd/2.0, Sd])      # 生成一个通道的mask
        mask = np.repeat(mask, c, axis=2)                                               # 在通道的维度复制，生成彩色的mask
        img[mask == 0] = 0                                                              # 椒
        img[mask == 1] = 255                                                            # 盐
        img= Image.fromarray(img.astype('uint8')).convert('RGB')                        # numpy转图片
        return img

def create_train_dataloader(root, use_flip, batch_size):
    '''
    Create train dataloader.
    root: the dataset root.
    use_flip: True or false.
    batch size: the batch size.
    '''
    main_trans_list = []
    if use_flip:
        main_trans_list.append(RandomHorizontalFlip())
    main_trans_list.append(PairedCrop())
    main_trans = Compose(main_trans_list)
    img_trans = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])])
    dataset = CrowdDataset(root=root, phase='train', main_transform=main_trans,
                           img_transform=img_trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def create_test_dataloader(root, batch_size):
    '''
    Create train dataloader.
    root: the dataset root.
    '''
    main_trans_list = []
    main_trans_list.append(PairedCrop())
    main_trans = Compose(main_trans_list)
    img_trans = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])])
    dataset = CrowdDataset(root=root, phase='test', main_transform=main_trans,
                           img_transform=img_trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


# ----------------------------------#
#          Transform code          #
# ----------------------------------#
def my_transform1(image):
    # 拿到角度的随机数。angle是一个-180到180之间的一个数
    angle = transforms.RandomRotation.get_params([-20, 20])
    # 对image和mask做相同的旋转操作，保证他们都旋转angle角度
    image = image.rotate(angle)
    return image

class RandomHorizontalFlip(object):
    '''
    Random horizontal flip.
    prob = 0.5
    '''

    def __call__(self, img_and_dmap):
        '''
        img: PIL.Image
        dmap: PIL.Image
        '''
        img = img_and_dmap
        img = my_transform1(img)
        if random.random() < 0.5:
            return (img.transpose(Image.FLIP_LEFT_RIGHT))
        else:
            return (img.transpose(Image.FLIP_TOP_BOTTOM))


class PairedCrop(object):
    '''
    Paired Crop for both image and its density map.
    Note that due to the maxpooling in the nerual network,
    we must promise that the size of input image is the corresponding factor.
    '''

    def __init__(self, factor=16):
        self.factor = factor

    @staticmethod
    def get_params(img, factor):
        w, h = img.size
        if w % factor == 0 and h % factor == 0:
            return 0, 0, h, w
        else:
            return 0, 0, h - (h % factor), w - (w % factor)

    def __call__(self, img_and_dmap):
        '''
        img: PIL.Image
        dmap: PIL.Image
        '''
        img = img_and_dmap
        return (img)

