import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([transforms.RandomCrop(96),transforms.ToTensor()])

class DataProcess(Dataset):
    def __init__(self,imgPath,transform = transform,ex=10):
        self.transforms = transform
        for _,_,files in os.walk(imgPath):
            self.imgs = [imgPath + file for file in files] * ex
        np.random.shuffle(self.imgs)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        t_img = self.imgs[item]
        t_img = Image.open(t_img)
        sourceImg = self.transforms(t_img)
        cropImg = torch.nn.MaxPool2d(4)(sourceImg)
        return cropImg,sourceImg

