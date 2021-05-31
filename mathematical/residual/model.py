from torch import nn
import torch

class BasicSRModel(nn.Module):
    def __init__(self, number_block):
        super(BasicSRModel, self).__init__()
        '''
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )
        '''
        self.number_block = number_block
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=None)
        self.first_block = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride=1, padding=1, groups=1, bias=True, padding_mode='zeros'),
            nn.LeakyReLU()
        )
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride=1, padding=1, groups=1, bias=True, padding_mode='zeros'),
            nn.LeakyReLU()
        )
        self.last_block = nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size = 3, stride=1, padding=1, groups=1, bias=True, padding_mode='zeros')


    def forward(self, x):
        '''
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
        '''
        #print(x.shape)
        x = self.up_sample(x)
        initial_x = x
        #print(x.shape)
        x = self.first_block(x)
        #print(x.shape)
        for i in range(self.number_block):
            x = self.blocks(x)
            #print(x.shape)
        x = self.last_block(x)
        #print(x.shape)
        x = initial_x + x
        return x

#model = NeuralNetwork().to(device)
#print(model)