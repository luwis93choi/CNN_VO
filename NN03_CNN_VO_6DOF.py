import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable

import numpy as np
from matplotlib import pyplot as plt
import math

class CNN_VO_6DOF(nn.Module):

    # Overriding base class of neural network (nn.Module)
    def __init__(self):
        super(CNN_VO_6DOF, self).__init__()

        self.use_cuda = False

        ###################################################################################################
        self.display_layer_list = ['']      ### Activate layer result display function by user's choice ###
        ###################################################################################################

        self.conv1 = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=(7, 7), stride=(3, 3), padding=(1, 1), bias=False)
        self.leakyrelu1 = nn.LeakyReLU(0.1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=7, stride=1)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), bias=False)
        self.leakyrelu2 = nn.LeakyReLU(0.1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=5, stride=1)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.leakyrelu3 = nn.LeakyReLU(0.1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=1)
        
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.leakyrelu4 = nn.LeakyReLU(0.1)

        self.fc_1 = nn.Linear(in_features = 512 * 5 * 24, out_features = 6)  # Fully Connected Layer 1

    # CNN Layer Result Display Function - Display 2D Convolution Results by Channels
    def layer_disp(self, conv_result_x):

        ####################################################################
        #### Fix this funciton to display the results from each channel ####
        ####################################################################    
        x_disp = conv_result_x.clone().detach().cpu()

        channel = x_disp.size(1)

        cols = 10
        rows = int(math.ceil(channel/cols))

        # for i in range(channel):
            
        #     plt.subplot(rows, cols, i+1)
        #     plt.imshow((x_disp.permute(0, 2, 3, 1)[0, :, :, i].numpy()*255).astype(np.uint8))

        plt.imshow((x_disp.permute(0, 2, 3, 1)[0, :, :, :3].numpy()*255).astype(np.uint8))
        plt.pause(0.001)
        plt.show(block=False)
        plt.clf()

    # Foward pass of DeepVO NN
    def forward(self, x):

        #self.layer_disp(x)
        x = self.conv1(x)
        x = self.leakyrelu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.leakyrelu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.leakyrelu3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.leakyrelu4(x)

        #print(x.size())   # Print the size of CNN output in order connect it to Fully Connected Layer
        # Reshpae/Flatten the output of common CNN
        x = x.view(x.size(0), 1, -1)

        # Forward pass into Linear Regression for pose estimation
        x = self.fc_1(x)

        #print(x.size())
        return x
