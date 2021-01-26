import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable

import numpy as np
from matplotlib import pyplot as plt
import math

import cv2 as cv

import time

class CNN_VO_Gray_Ensemble(nn.Module):

    # Overriding base class of neural network (nn.Module)
    def __init__(self):
        super(CNN_VO_Gray_Ensemble, self).__init__()

        self.use_cuda = False

        #################################
        # Ensemble 1 (High Filter Size) #
        #################################
        self.conv_E1_1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self.batch_norm_E1_1 = nn.BatchNorm2d(16)
        self.leakyrelu_E1_1 = nn.LeakyReLU(0.1)

        self.conv_E1_1_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        self.batch_norm_E1_1_1 = nn.BatchNorm2d(32)
        self.leakyrelu_E1_1_1 = nn.LeakyReLU(0.1)

        self.conv_E1_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.batch_norm_E1_2 = nn.BatchNorm2d(64)
        self.leakyrelu_E1_2 = nn.LeakyReLU(0.1)

        self.conv_E1_2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        self.batch_norm_E1_2_1 = nn.BatchNorm2d(64)
        self.leakyrelu_E1_2_1 = nn.LeakyReLU(0.1)

        # self.maxpool_E1_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_E1_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.batch_norm_E1_3 = nn.BatchNorm2d(128)
        self.leakyrelu_E1_3 = nn.LeakyReLU(0.1)

        self.conv_E1_3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        self.batch_norm_E1_3_1 = nn.BatchNorm2d(128)
        self.leakyrelu_E1_3_1 = nn.LeakyReLU(0.1)

        self.fc_E1_1 = nn.Linear(in_features = 128 * 24 * 80, out_features = 500)  
        self.layernorm_E1_1 = nn.LayerNorm(500)
        self.leakyrelu_fc_E1_1 = nn.LeakyReLU(0.1)
        self.dropoutE1_1 = nn.Dropout(p=0.5)

        self.fc_E1_2 = nn.Linear(in_features = 500, out_features = 1)  
        
        ###################################
        # Ensemble 2 (Medium Filter Size) #
        ###################################
        self.conv_E2_1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self.batch_norm_E2_1 = nn.BatchNorm2d(16)
        self.leakyrelu_E2_1 = nn.LeakyReLU(0.1)

        self.conv_E2_1_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        self.batch_norm_E2_1_1 = nn.BatchNorm2d(32)
        self.leakyrelu_E2_1_1 = nn.LeakyReLU(0.1)

        self.conv_E2_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.batch_norm_E2_2 = nn.BatchNorm2d(64)
        self.leakyrelu_E2_2 = nn.LeakyReLU(0.1)

        self.conv_E2_2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        self.batch_norm_E2_2_1 = nn.BatchNorm2d(64)
        self.leakyrelu_E2_2_1 = nn.LeakyReLU(0.1)

        # self.maxpool_E2_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_E2_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.batch_norm_E2_3 = nn.BatchNorm2d(128)
        self.leakyrelu_E2_3 = nn.LeakyReLU(0.1)

        self.conv_E2_3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        self.batch_norm_E2_3_1 = nn.BatchNorm2d(128)
        self.leakyrelu_E2_3_1 = nn.LeakyReLU(0.1)

        self.fc_E2_1 = nn.Linear(in_features = 128 * 24 * 80, out_features = 500)  
        self.layernorm_E2_1 = nn.LayerNorm(500)
        self.leakyrelu_fc_E2_1 = nn.LeakyReLU(0.1)
        self.dropoutE2_1 = nn.Dropout(p=0.5)
        
        self.fc_E2_2 = nn.Linear(in_features = 500, out_features = 1)  

        ##################################
        # Ensemble 3 (Small Filter Size) #
        ##################################
        self.conv_E3_1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self.batch_norm_E3_1 = nn.BatchNorm2d(16)
        self.leakyrelu_E3_1 = nn.LeakyReLU(0.1)

        self.conv_E3_1_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        self.batch_norm_E3_1_1 = nn.BatchNorm2d(32)
        self.leakyrelu_E3_1_1 = nn.LeakyReLU(0.1)

        self.conv_E3_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.batch_norm_E3_2 = nn.BatchNorm2d(64)
        self.leakyrelu_E3_2 = nn.LeakyReLU(0.1)

        self.conv_E3_2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        self.batch_norm_E3_2_1 = nn.BatchNorm2d(64)
        self.leakyrelu_E3_2_1 = nn.LeakyReLU(0.1)

        # self.maxpool_E3_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_E3_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.batch_norm_E3_3 = nn.BatchNorm2d(128)
        self.leakyrelu_E3_3 = nn.LeakyReLU(0.1)

        self.conv_E3_3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        self.batch_norm_E3_3_1 = nn.BatchNorm2d(128)
        self.leakyrelu_E3_3_1 = nn.LeakyReLU(0.1)

        self.fc_E3_1 = nn.Linear(in_features = 128 * 24 * 80, out_features = 500)  
        self.layernorm_E3_1 = nn.LayerNorm(500)
        self.leakyrelu_fc_E3_1 = nn.LeakyReLU(0.1)
        self.dropoutE3_1 = nn.Dropout(p=0.5)
        
        self.fc_E3_2 = nn.Linear(in_features = 500, out_features = 1)  

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

    # Foward pass of DeepVO NN
    def forward(self, x):

        # self.layer_disp(x, window_name='Input Img', col_num=2)
        # time.sleep(5)

        ######################
        # Ensemble 1 Forward #
        ######################

        x_E1 = self.conv_E1_1(x)
        x_E1 = self.batch_norm_E1_1(x_E1)
        x_E1 = self.leakyrelu_E1_1(x_E1)

        x_E1 = self.conv_E1_1_1(x_E1)
        x_E1 = self.batch_norm_E1_1_1(x_E1)
        x_E1 = self.leakyrelu_E1_1_1(x_E1)

        x_E1 = self.conv_E1_2(x_E1)
        x_E1 = self.batch_norm_E1_2(x_E1)
        x_E1 = self.leakyrelu_E1_2(x_E1)

        x_E1 = self.conv_E1_2_1(x_E1)
        x_E1 = self.batch_norm_E1_2_1(x_E1)
        x_E1 = self.leakyrelu_E1_2_1(x_E1)

        x_E1 = self.conv_E1_3(x_E1)
        x_E1 = self.batch_norm_E1_3(x_E1)
        x_E1 = self.leakyrelu_E1_3(x_E1)

        x_E1 = self.conv_E1_3_1(x_E1)
        x_E1 = self.batch_norm_E1_3_1(x_E1)
        x_E1 = self.leakyrelu_E1_3_1(x_E1)
        
        # print(x_E1.size())   # Print the size of CNN output in order connect it to Fully Connected Layer
        # Reshpae/Flatten the output of common CNN
        x_E1 = x_E1.reshape(x_E1.size(0), -1)

        # Forward pass into Linear Regression for pose estimation
        x_E1 = self.fc_E1_1(x_E1)
        x_E1 = self.layernorm_E1_1(x_E1)
        x_E1 = self.leakyrelu_fc_E1_1(x_E1)
        x_E1 = self.dropoutE1_1(x_E1)

        x_E1 = self.fc_E1_2(x_E1)
        
        ######################
        # Ensemble 2 Forward #
        ######################

        x_E2 = self.conv_E2_1(x)
        x_E2 = self.batch_norm_E2_1(x_E2)
        x_E2 = self.leakyrelu_E2_1(x_E2)

        x_E2 = self.conv_E2_1_1(x_E2)
        x_E2 = self.batch_norm_E2_1_1(x_E2)
        x_E2 = self.leakyrelu_E2_1_1(x_E2)

        x_E2 = self.conv_E2_2(x_E2)
        x_E2 = self.batch_norm_E2_2(x_E2)
        x_E2 = self.leakyrelu_E2_2(x_E2)

        x_E2 = self.conv_E2_2_1(x_E2)
        x_E2 = self.batch_norm_E2_2_1(x_E2)
        x_E2 = self.leakyrelu_E2_2_1(x_E2)

        x_E2 = self.conv_E2_3(x_E2)
        x_E2 = self.batch_norm_E2_3(x_E2)
        x_E2 = self.leakyrelu_E2_3(x_E2)

        x_E2 = self.conv_E2_3_1(x_E2)
        x_E2 = self.batch_norm_E2_3_1(x_E2)
        x_E2 = self.leakyrelu_E2_3_1(x_E2)

        # print(x_E2.size())   # Print the size of CNN output in order connect it to Fully Connected Layer
        # Reshpae/Flatten the output of common CNN
        x_E2 = x_E2.reshape(x_E2.size(0), -1)

        # Forward pass into Linear Regression for pose estimation
        x_E2 = self.fc_E2_1(x_E2)
        x_E2 = self.layernorm_E2_1(x_E2)
        x_E2 = self.leakyrelu_fc_E2_1(x_E2)
        x_E2 = self.dropoutE2_1(x_E2)

        x_E2 = self.fc_E2_2(x_E2)

        ######################
        # Ensemble 3 Forward #
        ######################

        x_E3 = self.conv_E3_1(x)
        x_E3 = self.batch_norm_E3_1(x_E3)
        x_E3 = self.leakyrelu_E3_1(x_E3)

        x_E3 = self.conv_E3_1_1(x_E3)
        x_E3 = self.batch_norm_E3_1_1(x_E3)
        x_E3 = self.leakyrelu_E3_1_1(x_E3)

        x_E3 = self.conv_E3_2(x_E3)
        x_E3 = self.batch_norm_E3_2(x_E3)
        x_E3 = self.leakyrelu_E3_2(x_E3)

        x_E3 = self.conv_E3_2_1(x_E3)
        x_E3 = self.batch_norm_E3_2_1(x_E3)
        x_E3 = self.leakyrelu_E3_2_1(x_E3)

        x_E3 = self.conv_E3_3(x_E3)
        x_E3 = self.batch_norm_E3_3(x_E3)
        x_E3 = self.leakyrelu_E3_3(x_E3)

        x_E3 = self.conv_E3_3_1(x_E3)
        x_E3 = self.batch_norm_E3_3_1(x_E3)
        x_E3 = self.leakyrelu_E3_3_1(x_E3)

        # print(x_E3.size())   # Print the size of CNN output in order connect it to Fully Connected Layer
        # Reshpae/Flatten the output of common CNN
        x_E3 = x_E3.reshape(x_E3.size(0), -1)

        # Forward pass into Linear Regression for pose estimation
        x_E3 = self.fc_E3_1(x_E3)
        x_E3 = self.layernorm_E3_1(x_E3)
        x_E3 = self.leakyrelu_fc_E3_1(x_E3)
        x_E3 = self.dropoutE3_1(x_E3)

        x_E3 = self.fc_E3_2(x_E3)

        return x_E1, x_E2, x_E3
