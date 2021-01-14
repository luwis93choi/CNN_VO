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
        self.batch_size = 1

        ###################################################################################################
        self.display_layer_list = ['']      ### Activate layer result display function by user's choice ###
        ###################################################################################################

        # self.dropout_input = nn.Dropout(p=0.1)

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.leakyrelu1 = nn.LeakyReLU(0.1)
        #self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=2)

        #self.dropout_c1 = nn.Dropout(p=0.5)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.leakyrelu2 = nn.LeakyReLU(0.1)
        #self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1)

        self.conv2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.batchnorm2_1 = nn.BatchNorm2d(128)
        self.leakyrelu2_1 = nn.LeakyReLU(0.1)
        self.maxpool2_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        #self.dropout_c2 = nn.Dropout(p=0.5)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.leakyrelu3 = nn.LeakyReLU(0.1)
        #self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=1)

        #self.dropout_c3 = nn.Dropout(p=0.5)

        self.conv3_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm3_1 = nn.BatchNorm2d(256)
        self.leakyrelu3_1 = nn.LeakyReLU(0.1)
        #self.maxpool3_1 = nn.MaxPool2d(kernel_size=3, stride=1)

        #self.dropout_c4 = nn.Dropout(p=0.5)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm4 = nn.BatchNorm2d(512)
        self.leakyrelu4 = nn.LeakyReLU(0.1)
        #self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv4_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm4_1 = nn.BatchNorm2d(512)
        self.leakyrelu4_1 = nn.LeakyReLU(0.1)
        #self.maxpool4_1 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm5 = nn.BatchNorm2d(1024)
        self.leakyrelu5 = nn.LeakyReLU(0.1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        #self.dropout0 = nn.Dropout(p=0.5)

        # self.linear1 = nn.Linear(in_features = 1024 * 12 * 40, out_features = 1000)
        # self.batchnorm_l1 = nn.BatchNorm1d(num_features=1)
        # self.leakyrelu_l1 = nn.LeakyReLU(0.1)
        # self.dropout1 = nn.Dropout(p=0.5)

        # self.linear2 = nn.Linear(in_features = 100, out_features = 50)
        # self.batchnorm_l2 = nn.BatchNorm1d(num_features=1)
        # self.leakyrelu_l2 = nn.LeakyReLU(0.1)
        # self.dropout2 = nn.Dropout(p=0.5)

        self.GRU_hidden_size = 300
        self.GRU_bidirection = True
        self.GRU_layer = 2
        self.GRU = nn.GRU(input_size=1024 * 6 * 20, hidden_size=self.GRU_hidden_size, num_layers=self.GRU_layer, bidirectional=self.GRU_bidirection, dropout=0.5, batch_first=True)

        self.fc = nn.Linear(in_features = self.GRU_hidden_size * 2, out_features = 6)  # Fully Connected Layer 1

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
            
            for j in range(cols):
                
                if (j + cols*i) >= channel:
                    blank = np.zeros((width, height, 1), np.uint8)
                    img_horizontal = cv.hconcat([img_horizontal, blank])

                elif j == 0:
                    img_horizontal = cv.resize(img_stack[:, :, j + cols*i], dsize=(0, 0), fx=resize_ratio, fy=resize_ratio, interpolation=cv.INTER_LINEAR)
                    width, height = img_horizontal.shape

                else:
                    input_img = cv.resize(img_stack[:, :, j + cols*i], dsize=(0, 0), fx=resize_ratio, fy=resize_ratio, interpolation=cv.INTER_LINEAR)
                    img_horizontal = cv.hconcat([img_horizontal, input_img])

            if i == 0:
                img_total = img_horizontal
            else:
                img_total = cv.vconcat([img_total, img_horizontal])

        cv.imshow(window_name, img_total)
        cv.waitKey(1)

    def init_hidden(self, batch_size=1):

        self.batch_size = batch_size

        if self.GRU_bidirection:
            return torch.zeros([self.GRU_layer * 2, batch_size, self.GRU_hidden_size], dtype=torch.float, requires_grad=True)
        else:
            return torch.zeros([self.GRU_layer, batch_size, self.GRU_hidden_size], dtype=torch.float, requires_grad=True)

    # Foward pass of DeepVO NN
    def forward(self, x, hidden_in):

        #self.layer_disp(x, 'Original', 2, 0.2)
        
        #x = self.dropout_input(x)
        
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.leakyrelu1(x)
        #x = self.maxpool1(x)
        #self.layer_disp(x, 'conv1', 5, 0.2)

        #x = self.dropout_c1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.leakyrelu2(x)
        #x = self.maxpool2(x)

        x = self.conv2_1(x)
        x = self.batchnorm2_1(x)
        x = self.leakyrelu2_1(x)
        x = self.maxpool2_1(x)

        #x = self.dropout_c2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.leakyrelu3(x)
        #x = self.maxpool3(x)

        x = self.conv3_1(x)
        x = self.batchnorm3_1(x)
        x = self.leakyrelu3_1(x)
        #x = self.maxpool3_1(x)

        #x = self.dropout_c3(x)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.leakyrelu4(x)
        #x = self.maxpool4(x)

        x = self.conv4_1(x)
        x = self.batchnorm4_1(x)
        x = self.leakyrelu4_1(x)
        #x = self.maxpool4_1(x)

        #x = self.dropout_c4(x)

        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.leakyrelu5(x)
        x = self.maxpool5(x)

        #self.layer_disp(x, 'conv5', 10, 2)

        #print(x.size())   # Print the size of CNN output in order connect it to Fully Connected Layer
        # Reshpae/Flatten the output of common CNN
        x = x.view(x.size(0), 1, -1)

        # x = self.dropout0(x)
        
        # x = self.linear1(x)
        # x = self.batchnorm_l1(x)
        # x = self.leakyrelu_l1(x)
        # x = self.dropout1(x)
        
        # x = self.linear2(x)
        # x = self.batchnorm_l2(x)
        # x = self.leakyrelu_l2(x)
        # x = self.dropout2(x)
        
        x, hidden_in = self.GRU(x, hidden_in.clone().detach())
        
        # Forward pass into Linear Regression for pose estimation
        x = self.fc(x)
        
        #print(x.size())
        return x, hidden_in
