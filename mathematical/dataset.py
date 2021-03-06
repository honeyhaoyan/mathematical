from __future__ import print_function, division
import os
import torch
#import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
from os import listdir
from os.path import isfile, join

class SRDataset(Dataset):

    def __init__(self, file_path, transform=None):
        
        #print("---------------------")
        #self.files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
        #print(self.files)
        datanames = os.listdir(file_path)
        files = []
        for dataname in datanames:
            if os.path.splitext(dataname)[1] == '.png':
                #print(dataname)
                files.append(join(file_path, dataname))
        self.files = files
        #print(files)

        self.path = file_path
        self.transform = transforms.Compose(
        [transforms.RandomCrop(64),
         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
         #transforms.Resize(size=(64,64),interpolation=2)
         ])
        self.low_resulotion = transforms.Resize(size=(32,32),interpolation=2)


    def __len__(self):
        #return len(self.landmarks_frame)
        return len(self.files)

    def __getitem__(self, idx):
        
        #print(idx)
        file_name = self.files[idx]
        #print(file_name)
        image = torchvision.io.read_image(file_name)
        #print(file_name)
        image = image.type(torch.FloatTensor)
        #print(image)
        image = image/255.0
        high_image = self.transform(image)
        low_image = self.low_resulotion(high_image)
        #print('--------------------------')
        #print(file_name)
        #print(image.shape)
        #print(low_image.shape)
        #print(image)
        #print(len(image))
        return low_image,high_image



'''
def load_data(root_path, dir, batch_size, phase):
    transform_dict = {
        'src': transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ]),
        'tar': transforms.Compose(
        [transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ])}
    data = datasets.ImageFolder(root=root_path + dir, transform=transform_dict[phase])
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    return data_loader 
'''
