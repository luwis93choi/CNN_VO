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

torch.manual_seed(42)
np.random.seed(42)

class CNN_VO_Gray(nn.Module):

    # Overriding base class of neural network (nn.Module)
    def __init__(self):
        super(CNN_VO_Gray, self).__init__()

        self.use_cuda = False

        #self.batch_norm0 = nn.BatchNorm2d(2)

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.leakyrelu1 = nn.LeakyReLU(0.1)

        self.conv1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        self.batch_norm1_1 = nn.BatchNorm2d(64)
        self.leakyrelu1_1 = nn.LeakyReLU(0.1)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.leakyrelu2 = nn.LeakyReLU(0.1)

        self.conv2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.batch_norm2_1 = nn.BatchNorm2d(128)
        self.leakyrelu2_1 = nn.LeakyReLU(0.1)
        
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        self.batch_norm2_2 = nn.BatchNorm2d(128)
        self.leakyrelu2_2 = nn.LeakyReLU(0.1)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.leakyrelu3 = nn.LeakyReLU(0.1)
        
        self.conv3_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch_norm3_1 = nn.BatchNorm2d(256)
        self.leakyrelu3_1 = nn.LeakyReLU(0.1)
        
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batch_norm3_2 = nn.BatchNorm2d(256)
        self.leakyrelu3_2 = nn.LeakyReLU(0.1)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch_norm4 = nn.BatchNorm2d(512)
        self.leakyrelu4 = nn.LeakyReLU(0.1)

        self.conv4_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch_norm4_1 = nn.BatchNorm2d(512)
        self.leakyrelu4_1 = nn.LeakyReLU(0.1)

        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batch_norm4_2 = nn.BatchNorm2d(512)
        self.leakyrelu4_2 = nn.LeakyReLU(0.1)

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batch_norm5 = nn.BatchNorm2d(1024)
        self.leakyrelu5 = nn.LeakyReLU(0.1)

        self.conv5_1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batch_norm5_1 = nn.BatchNorm2d(1024)
        self.leakyrelu5_1 = nn.LeakyReLU(0.1)

        self.fc_1 = nn.Linear(in_features = 1024 * 6 * 20, out_features = 100)  # Fully Connected Layer 1
        self.layernorm_1 = nn.LayerNorm(100)
        self.dropout1 = nn.Dropout(p=0.6)

        self.fc_2 = nn.Linear(in_features = 100, out_features = 100)  # Fully Connected Layer 2
        self.layernorm_2 = nn.LayerNorm(100)
        self.dropout2 = nn.Dropout(p=0.6)

        self.fc_3 = nn.Linear(in_features = 100, out_features = 100)  # Fully Connected Layer 2
        self.layernorm_3 = nn.LayerNorm(100)
        self.dropout3 = nn.Dropout(p=0.6)

        self.fc_4 = nn.Linear(in_features = 100, out_features = 100)  # Fully Connected Layer 2
        self.layernorm_4 = nn.LayerNorm(100)
        self.dropout4 = nn.Dropout(p=0.6)

        self.fc_5 = nn.Linear(in_features = 100, out_features = 3)  # Fully Connected Layer 2


        ### Weight Initialization ###
        # Fully Connected Layer weight init
        # torch.nn.init.xavier_normal_(self.conv1.weight)
        # torch.nn.init.xavier_normal_(self.conv1_1.weight)
        # torch.nn.init.xavier_normal_(self.conv2.weight)
        # torch.nn.init.xavier_normal_(self.conv2_1.weight)
        # torch.nn.init.xavier_normal_(self.conv2_2.weight)
        # torch.nn.init.xavier_normal_(self.conv3.weight)
        # torch.nn.init.xavier_normal_(self.conv3_1.weight)
        # torch.nn.init.xavier_normal_(self.conv3_2.weight)
        # torch.nn.init.xavier_normal_(self.conv4.weight)
        # torch.nn.init.xavier_normal_(self.conv4_1.weight)
        # torch.nn.init.xavier_normal_(self.conv4_2.weight)
        # torch.nn.init.xavier_normal_(self.conv5.weight)
        # torch.nn.init.xavier_normal_(self.conv5_1.weight)
        # torch.nn.init.xavier_normal_(self.fc_1.weight)
        # torch.nn.init.xavier_normal_(self.fc_2.weight)
        # torch.nn.init.xavier_normal_(self.fc_3.weight)
        # torch.nn.init.xavier_normal_(self.fc_4.weight)
        # torch.nn.init.xavier_normal_(self.fc_5.weight)

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
        # time.sleep(2)
        # x = self.batch_norm0(x)

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.leakyrelu1(x)

        x = self.conv1_1(x)
        x = self.batch_norm1_1(x)
        x = self.leakyrelu1_1(x)

        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.leakyrelu2(x)

        x = self.conv2_1(x)
        x = self.batch_norm2_1(x)
        x = self.leakyrelu2_1(x)
        
        x = self.conv2_2(x)
        x = self.batch_norm2_2(x)
        x = self.leakyrelu2_2(x)

        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.leakyrelu3(x)

        x = self.conv3_1(x)
        x = self.batch_norm3_1(x)
        x = self.leakyrelu3_1(x)

        x = self.conv3_2(x)
        x = self.batch_norm3_2(x)
        x = self.leakyrelu3_2(x)

        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.leakyrelu4(x)

        x = self.conv4_1(x)
        x = self.batch_norm4_1(x)
        x = self.leakyrelu4_1(x)

        x = self.conv4_2(x)
        x = self.batch_norm4_2(x)
        x = self.leakyrelu4_2(x)

        x = self.maxpool4(x)

        x = self.conv5(x)
        x = self.batch_norm5(x)
        x = self.leakyrelu5(x)

        x = self.conv5_1(x)
        x = self.batch_norm5_1(x)
        x = self.leakyrelu5_1(x)

        #print(x.size())   # Print the size of CNN output in order connect it to Fully Connected Layer
        # Reshpae/Flatten the output of common CNN
        x = x.reshape(x.size(0), -1)

        # Forward pass into Linear Regression for pose estimation
        x = self.fc_1(x)
        x = self.layernorm_1(x)
        x = self.dropout1(x)

        x = self.fc_2(x)
        x = self.layernorm_2(x)
        x = self.dropout2(x)

        x = self.fc_3(x)
        x = self.layernorm_3(x)
        x = self.dropout3(x)

        x = self.fc_4(x)
        x = self.layernorm_4(x)
        x = self.dropout4(x)

        x = self.fc_5(x)

        return x
