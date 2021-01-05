import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable

import numpy as np
from matplotlib import pyplot as plt

class Branched_CNN(nn.Module):

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
        super(Branched_CNN, self).__init__()

        self.use_cuda = False

        ###################################################################################################
        self.display_layer_list = ['']      ### Activate layer result display function by user's choice ###
        ###################################################################################################

        # Batch Normalization Preprocessing Layer
        #self.batch_norm0 = nn.BatchNorm2d(6)

        # Common CNN Layer 1
        self.conv1 = self.conv_layer('1', 6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), dropout_rate=0.5, use_batchNorm=True, use_Activation=True)

        # Common CNN Layer 2
        self.conv2 = self.conv_layer('2', 64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), dropout_rate=0.5, use_batchNorm=True, use_Activation=True)

        # Common CNN Layer 3
        self.conv3 = self.conv_layer('3', 128, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), dropout_rate=0.5, use_batchNorm=True, use_Activation=True)

        self.conv4_T = self.conv_layer('4_1', 256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dropout_rate=0.5, use_batchNorm=True, use_Activation=True)   # Translation CNN Layer 4
        self.conv5_T = self.conv_layer('5_1', 512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dropout_rate=0.5, use_batchNorm=True, use_Activation=True)  # Translation CNN Layer 5
        self.fc_T = nn.Linear(in_features = 160 * 48 * 1024, out_features = 3)  # Fully Connected Layer for Translation
        
        self.conv4_R = self.conv_layer('4_2', 256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dropout_rate=0.5, use_batchNorm=True, use_Activation=True)   # Rotation CNN Layer 4   
        self.conv5_R = self.conv_layer('5_2', 512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dropout_rate=0.5, use_batchNorm=True, use_Activation=True)  # Rotation CNN Layer 5
        self.fc_R = nn.Linear(in_features = 160 * 48 * 1024, out_features = 3)  # Fully Connected Layer for Rotation

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

        # Forward pass through Common CNN Layer 2
        x = self.conv2(x)

        # Forward pass through Common CNN Layer 3
        x = self.conv3(x)
        
        # Forward pass through Translation Layer
        x_T = self.conv4_T(x)
        x_T = self.conv5_T(x_T)

        # Reshpae/Flatten the output of common CNN
        x_T = x_T.view(x_T.size(0), -1)
        
        # Forward pass into Linear Regression in order to change output vectors into Translation vector
        x_T = self.fc_T(x_T)

        # Forward pass through Rotation Layer
        x_R = self.conv4_R(x)
        x_R = self.conv5_R(x_R)

        # Reshpae/Flatten the output of common CNN
        x_R = x_R.view(x_R.size(0), -1)

        # Forward pass into Linear Regression in order to change output vectors into Rotation vector
        x_R = self.fc_R(x_R)

        return x_T, x_R
