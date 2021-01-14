import os
import os.path
import numpy as np
import random
import math
import datetime

from PIL import Image   # Load images from the dataset

import torch
import torch.utils.data
import torchvision.transforms as tranforms
from matplotlib import pyplot as plt

class voDataLoader(torch.utils.data.Dataset):

    def __init__(self, img_dataset_path, pose_dataset_path, 
                       transform=None,  
                       sequence='00', batch_size=1):

        self.img_dataset_path = img_dataset_path
        self.pose_dataset_path = pose_dataset_path

        self.sequence = sequence

        self.data_idx = 0

        self.batch_size = batch_size

        self.len = 0    # The size of dataset in use

        self.transform = transform      # Image transformation conditions (Resolution Change)

        self.current_sequence_data_num = len(sorted(os.listdir(self.img_dataset_path + '/' + self.sequence + '/image_2')))
        self.img_path = sorted(os.listdir(self.img_dataset_path + '/' + self.sequence + '/image_2'))

        # Read 0th pose data and save it as current pose value
        self.pose_file = open(self.pose_dataset_path + '/' + self.sequence + '.txt', 'r')
        line = self.pose_file.readline()
        pose = line.strip().split()
        self.pose_file.close()

        self.current_pose_T = np.array([float(pose[3]), float(pose[7]), float(pose[11])])
        self.current_pose_Rmat = np.array([[float(pose[0]), float(pose[1]), float(pose[2])], 
                                            [float(pose[4]), float(pose[5]), float(pose[6])], 
                                            [float(pose[8]), float(pose[9]), float(pose[10])]])
        self.prev_pose_T = np.array([0.0, 0.0, 0.0])
        self.prev_pose_Rmat = np.array([0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0])
    
        self.lines = []
        f = open(self.pose_dataset_path + '/' + self.sequence + '.txt', 'r')
        while True:
            line = f.readline()
            self.lines.append(line)
            if not line: break

            self.len += 1
        self.len -= 1
        f.close()

        print('Sequence in Use : {}'.format(self.sequence))
        print('Size of dataset : {}'.format(self.len))

    # Dataset Load Function
    def __getitem__(self, index):

        index += 1

        if index >= 1:
            
            ### Dataset Image Preparation ###
            # Load Image at t-1 and t
            base_path = self.img_dataset_path + '/' + self.sequence + '/image_2'
            
            #pprev_img = Image.open(base_path + '/' + self.img_path[index-2]).convert('RGB')
            prev_img = Image.open(base_path + '/' + self.img_path[index-1]).convert('RGB')
            current_img = Image.open(base_path + '/' + self.img_path[index]).convert('RGB')

            # Transform the image according to the transformation conditions
            if self.transform is not None:
                #pprev_img = self.transform(pprev_img)
                prev_img = self.transform(prev_img)
                current_img = self.transform(current_img)

            ### Pose Data (Pose difference/change between t-1 and t) Preparation ###
            # Save previous groundtruth as t-1 value
            line = self.lines[index-1]
            pose = line.strip().split()
            self.prev_pose_T = np.array([float(pose[3]), float(pose[7]), float(pose[11])])
            self.prev_pose_Rmat = np.array([[float(pose[0]), float(pose[1]), float(pose[2])], 
                                            [float(pose[4]), float(pose[5]), float(pose[6])], 
                                            [float(pose[8]), float(pose[9]), float(pose[10])]])

            # Load groundtruth at t
            line = self.lines[index]
            pose = line.strip().split()
            self.current_pose_T = np.array([float(pose[3]), float(pose[7]), float(pose[11])])
            self.current_pose_Rmat = np.array([[float(pose[0]), float(pose[1]), float(pose[2])], 
                                               [float(pose[4]), float(pose[5]), float(pose[6])], 
                                               [float(pose[8]), float(pose[9]), float(pose[10])]])

            # Convert rotation matrix of groundtruth into euler angle
            prev_roll = math.atan2(self.prev_pose_Rmat[2][1], self.prev_pose_Rmat[2][2])
            prev_pitch = math.atan2(-1 * self.prev_pose_Rmat[2][0], math.sqrt(self.prev_pose_Rmat[2][1]**2 + self.prev_pose_Rmat[2][2]**2))
            prev_yaw = math.atan2(self.prev_pose_Rmat[1][0], self.prev_pose_Rmat[0][0])

            current_roll = math.atan2(self.current_pose_Rmat[2][1], self.current_pose_Rmat[2][2])
            current_pitch = math.atan2(-1 * self.current_pose_Rmat[2][0], math.sqrt(self.current_pose_Rmat[2][1]**2 + self.current_pose_Rmat[2][2]**2))
            current_yaw = math.atan2(self.current_pose_Rmat[1][0], self.current_pose_Rmat[0][0])

            # Compute the euler angle difference between groundtruth at t-1 and t
            droll = current_roll - prev_roll
            dpitch = current_pitch - prev_pitch
            dyaw = current_yaw - prev_yaw

            # Compute translation difference between groundtruth at t-1 and t
            dx = self.current_pose_T[0] - self.prev_pose_T[0]
            dy = self.current_pose_T[1] - self.prev_pose_T[1]
            dz = self.current_pose_T[2] - self.prev_pose_T[2]

            #########################################################################

            # Stack the image as indicated in DeepVO paper
            prev_current_stacked_img = np.asarray(np.concatenate([prev_img, current_img], axis=0))
            #prev_current_stacked_img = torch.Tensor.float(torch.from_numpy(np.concatenate([prev_img, current_img], axis=0)))

            # Prepare 6 DOF pose vector between t-1 and t (dX dY dZ dRoll dPitch dYaw)
            prev_current_odom = np.asarray([[dx, dy, dz, droll, dpitch, dyaw]])
            #prev_current_odom = torch.Tensor.float(torch.from_numpy(np.asarray([self.current_pose_T - self.prev_pose_T])))
            #prev_current_odom = torch.Tensor.float(torch.from_numpy(np.asarray([[dx, dy, dz, droll, dpitch, dyaw]])))
            
            #print(prev_current_stacked_img.shape)
            #print(prev_current_odom.shape)

            return prev_current_stacked_img, prev_current_odom
            
        else:

            print('[Dataloader] Index 0, 1 Skip')

            return [], []

    def __len__(self):

        return self.len-1
