import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable

import numpy as np
from matplotlib import pyplot as plt

class Branched_CNN(nn.Module):

    # Stages of Convolutional Layer
    def conv_layer(self, layer_num, in_channel, out_channel, kernel_size, stride, padding, dropout_rate=0.0, use_batchNorm=True, use_Activation=True):

        conv = nn.Sequential()
        conv.add_module('conv_'+layer_num, nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))

        if use_Activation:
            conv.add_module('LeakyReLU_'+layer_num, nn.LeakyReLU(0.1))

        if use_batchNorm:
            conv.add_module('batchNorm_'+layer_num, nn.BatchNorm2d(out_channel))

        if dropout_rate > 0.0:
            conv.add_module('dropout_'+layer_num, nn.Dropout(dropout_rate))

        return conv

    # Overriding base class of neural network (nn.Module)
    def __init__(self):
        super(Branched_CNN, self).__init__()

        self.use_cuda = False

        ###################################################################################################
        self.display_layer_list = ['']      ### Activate layer result display function by user's choice ###
        ###################################################################################################

        # Batch Normalization Preprocessing Layer
        # self.batch_norm0 = nn.BatchNorm2d(6)

        self.conv1 = self.conv_layer('1', 6, 6, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), dropout_rate=0.5, use_batchNorm=False, use_Activation=True)
        self.avgpool1 = nn.AvgPool2d(kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1)

        self.conv2 = self.conv_layer('2', 6, 30, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1), dropout_rate=0.5, use_batchNorm=False, use_Activation=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1)

        self.conv3 = self.conv_layer('3', 30, 30, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dropout_rate=0.5, use_batchNorm=False, use_Activation=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=1)
        
        self.conv4 = self.conv_layer('4', 30, 150, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dropout_rate=0.5, use_batchNorm=False, use_Activation=True)
        
        self.fc_1 = nn.Linear(in_features = 205 * 56 * 150, out_features = 12)  # Fully Connected Layer 1
        self.fc_2 = nn.Linear(in_features = 12, out_features = 6)  # Fully Connected Layer 2
        
        
    # CNN Layer Result Display Function - Display 2D Convolution Results by Channels
    def layer_disp(self, conv_result_x):

        ####################################################################
        #### Fix this funciton to display the results from each channel ####
        ####################################################################    
        x_disp = conv_result_x.clone().detach()
        plt.imshow((x_disp.permute(0, 2, 3, 1)[0, :, :, :3].cpu().numpy()*255).astype(np.uint8))
        plt.pause(0.001)
        plt.show(block=False)
        plt.clf()

    # Foward pass of DeepVO NN
    def forward(self, x):
        
        # Batch Normalization Preprocessing Layer
        #x = self.batch_norm0(x)

        # Forward pass through Common CNN Layer 1
        x = self.conv1(x)
        x = self.avgpool1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        # Forward pass through Common CNN Layer 2
        x = self.conv3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        #self.layer_disp(x)

        # print(x.size())   # Print the size of CNN output in order connect it to Fully Connected Layer
        # Reshpae/Flatten the output of common CNN
        x = x.view(x.size(0), -1)

        # Forward pass into Linear Regression for pose estimation
        x = self.fc_1(x)
        x = self.fc_2(x)

        return x
