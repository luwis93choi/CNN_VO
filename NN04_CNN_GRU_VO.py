import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable

import numpy as np
from matplotlib import pyplot as plt
import math

class CNN_GRU(nn.Module):

    # Overriding base class of neural network (nn.Module)
    def __init__(self):
        super(CNN_GRU, self).__init__()

        self.use_cuda = False

        ###################################################################################################
        self.display_layer_list = ['']      ### Activate layer result display function by user's choice ###
        ###################################################################################################

        self.conv1 = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(64, momentum=0.5)
        self.leakyrelu1 = nn.LeakyReLU(0.1)
        #self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.dropout1 = nn.Dropout(p=0.2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2 = nn.BatchNorm2d(128, momentum=0.5)
        self.leakyrelu2 = nn.LeakyReLU(0.1)
        #self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.dropout2 = nn.Dropout(p=0.2)

        self.conv2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2_1 = nn.BatchNorm2d(128, momentum=0.5)
        self.leakyrelu2_1 = nn.LeakyReLU(0.1)
        #self.maxpool2_1 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.dropout2_1 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(256, momentum=0.5)
        self.leakyrelu3 = nn.LeakyReLU(0.1)
        #self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.dropout3 = nn.Dropout(p=0.2)

        self.conv3_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm3_1 = nn.BatchNorm2d(256, momentum=0.5)
        self.leakyrelu3_1 = nn.LeakyReLU(0.1)
        #self.maxpool3_1 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        self.dropout3_1 = nn.Dropout(p=0.2)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm4 = nn.BatchNorm2d(512, momentum=0.5)
        self.leakyrelu4 = nn.LeakyReLU(0.1)

        self.dropout4 = nn.Dropout(p=0.2)

        self.GRU = nn.GRU(input_size=512 * 12 * 40, hidden_size=100, num_layers=2, bidirectional=True, dropout=0.2, batch_first=True)

        self.fc = nn.Linear(in_features = 200, out_features = 3)  # Fully Connected Layer 1

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

    def init_hidden(self):
        
        return torch.zeros([4, 1, 100], dtype=torch.float, requires_grad=True)

    # Foward pass of DeepVO NN
    def forward(self, x, hidden_in):

        #self.layer_disp(x)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.leakyrelu1(x)
        #x = self.maxpool1(x)

        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.leakyrelu2(x)
        #x = self.maxpool2(x)

        x = self.dropout2(x)

        x = self.conv2_1(x)
        x = self.batchnorm2_1(x)
        x = self.leakyrelu2_1(x)
        #x = self.maxpool2_1(x)

        x = self.dropout2_1(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.leakyrelu3(x)
        #x = self.maxpool3(x)

        x = self.dropout3(x)

        x = self.conv3_1(x)
        x = self.batchnorm3_1(x)
        x = self.leakyrelu3_1(x)
        #x = self.maxpool3_1(x)

        x = self.dropout3_1(x)

        x = self.conv4(x)
        x = self.leakyrelu4(x)

        x = self.dropout4(x)

        #print(x.size())   # Print the size of CNN output in order connect it to Fully Connected Layer
        # Reshpae/Flatten the output of common CNN
        x = x.view(x.size(0), 1, -1)
        
        x, hidden_out = self.GRU(x, hidden_in.clone().detach())
        
        # Forward pass into Linear Regression for pose estimation
        x = self.fc(x)
        
        #print(x.size())
        return x, hidden_out
