import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable

import numpy as np
from matplotlib import pyplot as plt
import math

import cv2 as cv

class CNN_GRU(nn.Module):

    # Overriding base class of neural network (nn.Module)
    def __init__(self):
        super(CNN_GRU, self).__init__()

        self.use_cuda = False

        ###################################################################################################
        self.display_layer_list = ['']      ### Activate layer result display function by user's choice ###
        ###################################################################################################

        self.dropout_input = nn.Dropout(p=0.1)

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(20, 20), stride=(1, 1), padding=(2, 2), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.leakyrelu1 = nn.LeakyReLU(0.1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.dropout_c1 = nn.Dropout(p=0.5)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(15, 15), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.leakyrelu2 = nn.LeakyReLU(0.1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.dropout_c2 = nn.Dropout(p=0.5)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(10, 10), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.leakyrelu3 = nn.LeakyReLU(0.1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.dropout_c3 = nn.Dropout(p=0.5)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.leakyrelu4 = nn.LeakyReLU(0.1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.dropout0 = nn.Dropout(p=0.5)

        self.linear1 = nn.Linear(in_features = 128 * 74 * 298, out_features = 100)
        self.batchnorm_l1 = nn.BatchNorm1d(num_features=1)
        self.leakyrelu_l1 = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(p=0.5)

        self.linear2 = nn.Linear(in_features = 100, out_features = 50)
        self.batchnorm_l2 = nn.BatchNorm1d(num_features=1)
        self.leakyrelu_l2 = nn.LeakyReLU(0.1)
        self.dropout2 = nn.Dropout(p=0.5)

        self.GRU_hidden_size = 50
        self.GRU = nn.GRU(input_size=50, hidden_size=self.GRU_hidden_size, num_layers=2, bidirectional=False, dropout=0.5, batch_first=True)

        self.fc = nn.Linear(in_features = self.GRU_hidden_size, out_features = 6)  # Fully Connected Layer 1

    # CNN Layer Result Display Function - Display 2D Convolution Results by Channels
    def layer_disp(self, conv_result_x, window_name, col_num, resize_ratio=0.8, invert=False):

        x_disp = conv_result_x.clone().detach().cpu()

        if invert == True:
            img_stack = 255 - (x_disp.permute(0, 2, 3, 1)[0, :, :, :].numpy()*255).astype(np.uint8)
        else:
            img_stack = (x_disp.permute(0, 2, 3, 1)[0, :, :, :].numpy()*255).astype(np.uint8)

        channel = img_stack.shape[2]

        cols = col_num
        rows = int(math.ceil(channel/cols))

        resize_ratio = resize_ratio

        for i in range(rows):
            
            img_horizontal = cv.resize(img_stack[:, :, cols*i], dsize=(0, 0), fx=resize_ratio, fy=resize_ratio, interpolation=cv.INTER_LINEAR)

            width, height = img_horizontal.shape

            for j in range(cols):
                
                if (j + cols*i) >= channel:
                    blank = np.zeros((width, height, 1), np.uint8)
                    img_horizontal = cv.hconcat([img_horizontal, blank])

                else:
                    input_img = cv.resize(img_stack[:, :, j + cols*i], dsize=(0, 0), fx=resize_ratio, fy=resize_ratio, interpolation=cv.INTER_LINEAR)
                    img_horizontal = cv.hconcat([img_horizontal, input_img])

            if i == 0:
                img_total = img_horizontal
            else:
                img_total = cv.vconcat([img_total, img_horizontal])

        cv.imshow(window_name, img_total)
        cv.waitKey(1)

    def init_hidden(self):
        
        return torch.zeros([2, 1, self.GRU_hidden_size], dtype=torch.float, requires_grad=True)

    # Foward pass of DeepVO NN
    def forward(self, x, hidden_in):

        #self.layer_disp(x, 'Original', 1, 0.2)
        #print(x.size())

        x = self.dropout_input(x)
        #self.layer_disp(x)

        #self.layer_disp(x)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.leakyrelu1(x)
        x = self.maxpool1(x)
        #self.layer_disp(x, 'conv1', 5, 0.5)

        x = self.dropout_c1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.leakyrelu2(x)
        x = self.maxpool2(x)

        x = self.dropout_c2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.leakyrelu3(x)
        x = self.maxpool3(x)

        x = self.dropout_c3(x)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.leakyrelu4(x)
        x = self.maxpool4(x)
        #self.layer_disp(x, 'conv4', 10, 0.8)

        #print(x.size())   # Print the size of CNN output in order connect it to Fully Connected Layer
        # Reshpae/Flatten the output of common CNN
        x = x.view(x.size(0), 1, -1)

        x = self.dropout0(x)
        
        x = self.linear1(x)
        x = self.batchnorm_l1(x)
        x = self.leakyrelu_l1(x)
        x = self.dropout1(x)
        
        x = self.linear2(x)
        x = self.batchnorm_l2(x)
        x = self.leakyrelu_l2(x)
        x = self.dropout2(x)
        
        x, hidden_in = self.GRU(x, hidden_in.detach())
        
        # Forward pass into Linear Regression for pose estimation
        x = self.fc(x)
        
        #print(x.size())
        return x, hidden_in
