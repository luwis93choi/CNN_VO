from dataloader_v2 import KITTI_Dataset

from notifier import notifier_Outlook

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable

from torchsummaryX import summary

import datetime
import time
import numpy as np
import math
from matplotlib import pyplot as plt
import sys
import os
import pickle

class tester():

    def __init__(self, NN_model=None, checkpoint=None,
                       model_path='./',
                       use_cuda=True, cuda_num='',
                       loader_preprocess_param=transforms.Compose([]), 
                       img_dataset_path='', pose_dataset_path='',
                       test_epoch=1, test_sequence=[], test_batch=1,
                       plot_epoch=True,
                       sender_email='', sender_email_pw='', receiver_email=''):

        self.NN_model = NN_model
        self.use_cuda = use_cuda
        self.cuda_num = cuda_num

        self.img_dataset_path = img_dataset_path
        self.pose_dataset_path = pose_dataset_path
        self.model_path = model_path

        self.test_epoch = test_epoch
        self.test_sequence = test_sequence
        self.test_batch = test_batch

        self.plot_epoch = plot_epoch

        self.sender_email = sender_email
        self.sender_pw = sender_email_pw
        self.receiver_email = receiver_email

        ### CUDA / CPU Preparation ###
        if (use_cuda == True) and (cuda_num != ''):        
            # Load main processing unit for neural network
            self.PROCESSOR = torch.device('cuda:'+self.cuda_num if torch.cuda.is_available() else 'cpu')

        else:
            self.PROCESSOR = torch.device('cpu')

        print(str(self.PROCESSOR))

        ### Model reloading ###
        if checkpoint == None:
            sys.exit('[Tester ERROR] Invalid checkpoint loading')

        elif checkpoint != None:

            if NN_model == None:

                sys.exit('[Tester ERROR] No NN model is specified')

            else:

                self.NN_model.to(self.PROCESSOR)
                self.NN_model.load_state_dict(checkpoint['model_state_dict'])

                self.model_path = './'

                print('Model state loaded')

        self.NN_model.eval()

        Test_KITTI_Dataset = KITTI_Dataset(name='KITTI_Test',
                                           img_dataset_path=self.img_dataset_path,
                                           pose_dataset_path=self.pose_dataset_path,
                                           transform=loader_preprocess_param,
                                           sequence=test_sequence, verbose=0)

        self.test_loader = torch.utils.data.DataLoader(Test_KITTI_Dataset, batch_size=self.test_batch, num_workers=8, shuffle=False, drop_last=True)

        self.translation_loss = nn.MSELoss()
        self.rotation_loss = nn.MSELoss()

        #summary(self.NN_model, (torch.zeros((1, 6, 384, 1280)).to(self.PROCESSOR)))

        # Prepare Email Notifier
        self.notifier = notifier_Outlook(sender_email=self.sender_email, sender_email_pw=self.sender_pw)

    def run_test(self):

        start_time = str(datetime.datetime.now())

        estimated_x = 0.0
        estimated_y = 0.0
        estimated_z = 0.0

        GT_x = 0.0
        GT_y = 0.0
        GT_z = 0.0

        current_pose_T = np.array([[0], 
                                   [0], 
                                   [0]])

        current_pose_R = np.array([[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]])
        test_loss = []

        test_loss_sum = 0.0
        test_T_loss_sum = 0.0
        test_R_loss_sum = 0.0
        test_length = 0
        
        fig = plt.figure(figsize=(20, 10))
        plt.grid(True)

        for epoch in range(self.test_epoch):
            
            # Prevent the model from updating gradients by adding 'torch.no_grad()' & not using optmizer steps
            with torch.no_grad():
                
                self.NN_model.eval()

                print('[EPOCH] : {}'.format(epoch))

                loss_sum = 0.0

                before_epoch = time.time()

                for batch_idx, (prev_current_img, prev_current_odom) in enumerate(self.test_loader):
                    

                    ### Data GPU Transfer ###
                    if self.use_cuda == True:
                        prev_current_img = prev_current_img.to(self.PROCESSOR)
                        prev_current_odom = prev_current_odom.to(self.PROCESSOR)

                    ### Model Prediction ###
                    estimated_pose_vect = self.NN_model(prev_current_img)
            
                    print('----------------------------------------------------------------------')
                    print(estimated_pose_vect)
                    print(prev_current_odom)

                    predicted_dx = estimated_pose_vect.clone().detach().cpu().numpy()[0][0]
                    predicted_dy = estimated_pose_vect.clone().detach().cpu().numpy()[0][1]
                    predicted_dz = estimated_pose_vect.clone().detach().cpu().numpy()[0][2]

                    ### VO Estimation Plotting ##
                    estimated_x = estimated_x + predicted_dx
                    estimated_z = estimated_z + predicted_dz

                    plt.plot(estimated_x, estimated_z, 'bo')

                    ### Groundtruth Plotting ###
                    GT = prev_current_odom.clone().detach().cpu().numpy()
                    GT_prev_current_x = GT[0][0]
                    GT_prev_current_y = GT[0][1]
                    GT_prev_current_z = GT[0][2]

                    GT_x = GT_x + GT_prev_current_x
                    GT_y = GT_y + GT_prev_current_y
                    GT_z = GT_z + GT_prev_current_z

                    plt.plot(GT_x, GT_z, 'ro')
                    plt.pause(0.001)
                    plt.show(block=False)

                    ### Loss Computation ###
                    
                    self.loss = self.translation_loss(estimated_pose_vect.float()[:, :3], prev_current_odom.float()[:, :3]) + 100 * self.rotation_loss(estimated_pose_vect.float()[:, 3:], prev_current_odom.float()[:, 3:])
                
                    ### Translation/Rotation Loss ###
                    T_loss = self.translation_loss(estimated_pose_vect.float()[:, :3], prev_current_odom.float()[:, :3]).item()
                    test_T_loss_sum += T_loss

                    R_loss = 100 * self.rotation_loss(estimated_pose_vect.float()[:, 3:], prev_current_odom.float()[:, 3:]).item()
                    test_R_loss_sum += T_loss
                    
                    ### Accumulate total loss ###
                    test_loss_sum += float(self.loss.item())
                    test_length += 1

                    updates = []
                    updates.append('\n')
                    updates.append('[Test Epoch {}/{}][Progress : {:.2%}][Batch Idx : {}] \n'.format(epoch, self.test_epoch, batch_idx/len(self.test_loader), batch_idx))
                    updates.append('Batch Loss : {:.4f} / Translation Loss : {:.4f} / Rotation Loss : {:.4f} \n'.format(self.loss.item(), T_loss, R_loss))
                    updates.append('Average Loss : {:.4f} / Avg Translation Loss : {:.4f} / Avg Rotation Loss : {:.4f} \n'.format(test_loss_sum/test_length, test_T_loss_sum/test_length, test_R_loss_sum/test_length))
                    final_updates = ''.join(updates)

                    sys.stdout.write(final_updates)

                    # if batch_idx < len(self.test_loader)-1:
                    #     for line_num in range(len(updates)):
                    #         sys.stdout.write("\x1b[1A\x1b[2K")

                after_epoch = time.time()