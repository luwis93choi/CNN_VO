import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable

import numpy as np
from matplotlib import pyplot as plt
import math

class Auto_CNN_VO(nn.Module):

    # Overriding base class of neural network (nn.Module)
    def __init__(self):
        super(Auto_CNN_VO, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(7, 7), stride=(2, 2), padding=(1, 1), bias=False)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), bias=False)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), bias=False)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.relu5 = nn.ReLU()

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.relu_d1 = nn.ReLU()
        
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.relu_d2 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
        self.relu_d3 = nn.ReLU()
        
        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
        self.relu_d4 = nn.ReLU()

        self.deconv5 = nn.ConvTranspose2d(in_channels=16, out_channels=6, kernel_size=(7, 7), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
        self.relu_d5 = nn.ReLU()

        # Pose Estimation Net
        self.conv6_pose = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(3, 3), bias=False)
        self.conv7_pose = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(3, 3), bias=False)
        self.conv8_pose = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=3, padding=1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 6))
        #self.fc = nn.Linear(in_features = 256 * 22 * 78, out_features = 6)

    def reset_weights_Xavier(self):

        for layer in self.children():
            
            if type(layer) == nn.Conv2d:
                print('Xavier Init : {}'.format(layer))
                nn.init.xavier_uniform_(layer.weight)
            elif type(layer) == nn.ConvTranspose2d:
                print('Xavier Init : {}'.format(layer))
                nn.init.xavier_uniform_(layer.weight)
            elif type(layer) == nn.Linear:
                print('Xavier Init : {}'.format(layer))
                nn.init.xavier_uniform_(layer.weight)

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

    def encoder(self, x):

        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = x.view(x.size(0), -1)

        return x 

    def decoder(self, x):

        x = x.view(x.size(0), 256, 22, 78)
        x = self.deconv1(x)
        x = self.relu_d1(x)

        x = self.deconv2(x)
        x = self.relu_d2(x)

        return x

    def pose_estimater(self, x):

        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x 

    # Foward pass
    def forward(self, x):
        
        #self.layer_disp(x)

        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)

        #x_pose = x.clone()
        x_pose = self.conv6_pose(x)
        x_pose = self.conv7_pose(x_pose)
        x_pose = x_pose.reshape(x.size(0), x.size(1), -1)
        x_pose = self.conv8_pose(x_pose)
        pose_est = self.avgpool(x_pose)
        #print(pose_est.size())
        
        x_decode = x.view(x.size(0), -1)

        x_decode = x_decode.view(x_decode.size(0), 256, 22, 78)
        x_decode = self.deconv1(x_decode)
        x_decode = self.relu_d1(x_decode)

        x_decode = self.deconv2(x_decode)
        x_decode = self.relu_d2(x_decode)
        
        x_decode = self.deconv3(x_decode)
        x_decode = self.relu_d3(x_decode)
        
        x_decode = self.deconv4(x_decode)
        x_decode = self.relu_d4(x_decode)
        
        x_decode = self.deconv5(x_decode)
        x_decode = self.relu_d5(x_decode)

        #self.layer_disp(x)
        
        return x_decode, pose_est