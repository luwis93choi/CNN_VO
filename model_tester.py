from dataloader import voDataLoader

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

                self.NN_model = NN_model
                self.NN_model.load_state_dict(checkpoint['model_state_dict'])

                self.model_path = './'

        self.NN_model.train()

        self.NN_model.to(self.PROCESSOR)

        self.test_loader_list = []
        for i in range(len(test_sequence)):
            self.test_loader_list.append(torch.utils.data.DataLoader(voDataLoader(img_dataset_path=self.img_dataset_path,
                                                                                   pose_dataset_path=self.pose_dataset_path,
                                                                                   transform=loader_preprocess_param,
                                                                                   sequence=test_sequence[i],
                                                                                   batch_size=self.test_batch),
                                                                                   batch_size=self.test_batch, shuffle=False, drop_last=True))

        self.translation_loss = nn.MSELoss()
        self.angular_loss = nn.MSELoss()

        self.pose_loss = nn.MSELoss()

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

        fig = plt.figure(figsize=(20, 10))
        plt.grid(True)

        for epoch in range(self.test_epoch):
            
            with torch.no_grad():

                print('[EPOCH] : {}'.format(epoch))

                loss_sum = 0.0

                before_epoch = time.time()

                for test_loader in self.test_loader_list:

                    print('-------')
                    for batch_idx, (sequence, prev_current_img, prev_current_odom) in enumerate(test_loader):
                            
                        if batch_idx == 0:
                            print('Index 0 Skip')

                        else:
                            ### Input Image Display ###
                            # plt.imshow((prev_current_img.permute(0, 2, 3, 1)[0, :, :, :3].cpu().numpy()*255).astype(np.uint8))
                            # plt.pause(0.001)
                            # plt.show(block=False)
                            # plt.clf()

                            ### Data GPU Transfer ###
                            if self.use_cuda == True:
                                prev_current_img = Variable(prev_current_img.to(self.PROCESSOR))
                                prev_current_odom = Variable(prev_current_odom.to(self.PROCESSOR))

                            ### Model Prediction ###
                            estimated_pose_vect = self.NN_model(prev_current_img)
                            
                            predicted_dx = estimated_pose_vect.data.cpu().numpy()[0][0]
                            predicted_dy = estimated_pose_vect.data.cpu().numpy()[0][1]
                            predicted_dz = estimated_pose_vect.data.cpu().numpy()[0][2]

                            predicted_roll = estimated_pose_vect.data.cpu().numpy()[0][3]
                            predicted_pitch = estimated_pose_vect.data.cpu().numpy()[0][4]
                            predicted_yaw = estimated_pose_vect.data.cpu().numpy()[0][5]

                            ### VO Estimation Plotting ##
                            estimated_x = estimated_x + predicted_dx
                            estimated_z = estimated_z + predicted_dz

                            plt.plot(estimated_x, estimated_z, 'bo')

                            ### Groundtruth Plotting ###
                            GT = prev_current_odom.data.cpu().numpy()
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
                            self.loss = self.pose_loss(estimated_pose_vect.float(), prev_current_odom.float())

                            ### Accumulate total loss ###
                            loss_sum += float(self.loss.item())

                            print('[Epoch {}/{}][Sequence : {}][batch_idx : {}] - Batch Loss : {} / Total Loss : {}'.format(epoch, self.test_epoch ,sequence, batch_idx, float(self.loss.item()), loss_sum))

                after_epoch = time.time()

                test_loss.append(loss_sum)

                if self.plot_epoch == True:
                    plt.clf()
                    plt.figure(figsize=(20, 8))
                    plt.plot(range(len(test_loss)), test_loss, 'bo-')
                    plt.title('CNN VO Test with KITTI [Total MSE Loss]\nTest Sequence ' + str(self.train_sequence))
                    plt.xlabel('Test Length')
                    plt.ylabel('Total Loss')
                    plt.savefig(self.model_path + 'Test ' + start_time + '.png')