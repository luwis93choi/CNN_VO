import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable

import numpy as np

class CNN_implicit_Param(nn.Module):

    # Stages of Convolutional Layer
    def conv_layer(self, layer_num, in_channel, out_channel, kernel_size, stride, padding, dropout_rate, use_batchNorm=True, use_Activation=True):

        conv = nn.Sequential()
        conv.add_module('conv'+layer_num, nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))

        if use_batchNorm:
            conv.add_module('batchNorm'+layer_num, nn.BatchNorm2d(out_channel))

        if use_Activation:
            conv.add_module('LeakyReLU'+layer_num, nn.LeakyReLU(0.1))

        conv.add_module('dropout'+layer_num, nn.Dropout(dropout_rate))

        return conv

    # DeepVO NN Initialization
    # Overriding base class of neural network (nn.Module)
    def __init__(self):
        super(CNN_implicit_Param, self).__init__()

        self.use_cuda = False

        # CNN Layer 1
        self.conv1 = self.conv_layer('1', 6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), dropout_rate=0.5, use_batchNorm=True, use_Activation=True)

        # CNN Layer 2
        self.conv2 = self.conv_layer('2', 64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), dropout_rate=0.5, use_batchNorm=True, use_Activation=True)

        # CNN Layer 3
        self.conv3 = self.conv_layer('3', 128, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), dropout_rate=0.5, use_batchNorm=True, use_Activation=True)

        # CNN Layer 3_1
        self.conv3_1 = self.conv_layer('3_1', 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dropout_rate=0.5, use_batchNorm=True, use_Activation=True)

        # CNN Layer 4
        self.conv4 = self.conv_layer('4', 256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dropout_rate=0.5, use_batchNorm=True, use_Activation=True)

        # CNN Layer 4_1
        self.conv4_1 = self.conv_layer('4_1', 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dropout_rate=0.5, use_batchNorm=True, use_Activation=True)

        # CNN Layer 5
        self.conv5 = self.conv_layer('5', 512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dropout_rate=0.5, use_batchNorm=True, use_Activation=True)

        # CNN Layer 5_1
        self.conv5_1 = self.conv_layer('5_1', 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dropout_rate=0.5, use_batchNorm=True, use_Activation=True)

        # CNN Layer 6
        self.conv6 = self.conv_layer('6', 512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dropout_rate=0.5, use_batchNorm=True, use_Activation=True)

        # Fully Connected Layer
        self.fc = nn.Linear(in_features = 6 * 20 * 1024, out_features = 6)

    # Foward pass of DeepVO NN
    def forward(self, x):
        
        # Forward pass through CNN Layer 1
        x = self.conv1(x)

        # Forward pass through CNN Layer 2
        x = self.conv2(x)

        # Forward pass through CNN Layer 3
        x = self.conv3(x)

        # Forward pass through CNN Layer 3_1
        x = self.conv3_1(x)

        # Forward pass through CNN Layer 4
        x = self.conv4(x)

        # Forward pass through CNN Layer 4_1
        x = self.conv4_1(x)

        # Forward pass through CNN Layer 5
        x = self.conv5(x)

        # Forward pass through CNN Layer 5_1
        x = self.conv5_1(x)

        # Foward pass through CNN Layer 6
        x = self.conv6(x)

        # Reshpae/Flatten the output of CNN in order to use it as the input of RNN
        x = x.view(x.size(0), -1)     # Input Shape : [Length of recurrent sequence, Size(0), Size(1)]

        #x = x.flatten()

        # Forward pass into Linear Regression in order to change output vectors into Pose vector
        x = self.fc(x)

        return x

    def get_pose_loss(self, _estimated_scale, _groundtruth_scale):

        # Custom Loss Function if necessary

        return ''