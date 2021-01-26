import os
import os.path
import numpy as np
import random
import math
import datetime

import csv

from PIL import Image   # Load images from the dataset

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

torch.manual_seed(42)
np.random.seed(42)

class KITTI_Dataset_gray(torch.utils.data.Dataset):

    def __init__(self, name='',
                       img_dataset_path='', pose_dataset_path='', 
                       train_transform=None,  
                       valid_transform=None,  
                       sequence=['00'], verbose=0, cm=False, mode='training'):

        self.name = name

        self.img_dataset_path = img_dataset_path
        self.pose_dataset_path = pose_dataset_path

        self.sequence = sequence

        self.data_idx = 0

        self.len = 0    # The size of dataset in use

        self.train_transform = train_transform
        self.valid_transform = valid_transform

        self.reader = None

        self.verbose = verbose

        self.cm = cm

        self.mode = mode
        
        self.dataset_dict = open('./' + self.name + '_dataset_dict.csv', 'w', encoding='utf-8', newline='')
        self.dataset_writer = csv.writer(self.dataset_dict)
        
        header_list = ['current_index', 'current_img_path', 'prev_img_path', 'current_pose_x', 'current_pose_y', 'current_pose_z', 
                                                                             'current_pose_roll', 'current_pose_pitch', 'current_pose_yaw', 
                                                                             'prev_pose_x', 'prev_pose_y', 'prev_pose_z', 
                                                                             'prev_pose_roll', 'prev_pose_pitch', 'prev_pose_yaw']
        self.dataset_writer.writerow(header_list)
        
        for sequence_idx in range(len(sequence)):

            base_path = self.img_dataset_path + '/' + self.sequence[sequence_idx] + '/image_2'

            img_name = sorted(os.listdir(self.img_dataset_path + '/' + self.sequence[sequence_idx] + '/image_2'))

            # Pose data accumulation
            lines = []
            pose_file = open(self.pose_dataset_path + '/' + self.sequence[sequence_idx] + '.txt', 'r')
            while True:
                line = pose_file.readline()
                lines.append(line)
                if not line: break
            pose_file.close()

            for data_idx in range(len(img_name)):

                if data_idx != 0:
            
                    # Save previous groundtruth as t-1 value
                    line = lines[data_idx-1]
                    pose = line.strip().split()
                    prev_pose_T = [float(pose[3]), float(pose[7]), float(pose[11])]
                    prev_pose_Rmat = np.array([[float(pose[0]), float(pose[1]), float(pose[2])], 
                                                    [float(pose[4]), float(pose[5]), float(pose[6])], 
                                                    [float(pose[8]), float(pose[9]), float(pose[10])]])

                    # Load groundtruth at t
                    line = lines[data_idx]
                    pose = line.strip().split()
                    current_pose_T = [float(pose[3]), float(pose[7]), float(pose[11])]
                    current_pose_Rmat = np.array([[float(pose[0]), float(pose[1]), float(pose[2])], 
                                                    [float(pose[4]), float(pose[5]), float(pose[6])], 
                                                    [float(pose[8]), float(pose[9]), float(pose[10])]])

                    # Convert rotation matrix of groundtruth into euler angle
                    prev_roll = math.atan2(prev_pose_Rmat[2][1], prev_pose_Rmat[2][2])
                    prev_pitch = math.atan2(-1 * prev_pose_Rmat[2][0], math.sqrt(prev_pose_Rmat[2][1]**2 + prev_pose_Rmat[2][2]**2))
                    prev_yaw = math.atan2(prev_pose_Rmat[1][0], prev_pose_Rmat[0][0])

                    current_roll = math.atan2(current_pose_Rmat[2][1], current_pose_Rmat[2][2])
                    current_pitch = math.atan2(-1 * current_pose_Rmat[2][0], math.sqrt(current_pose_Rmat[2][1]**2 + current_pose_Rmat[2][2]**2))
                    current_yaw = math.atan2(current_pose_Rmat[1][0], current_pose_Rmat[0][0])

                    data = [data_idx, base_path + '/' + img_name[data_idx], base_path + '/' + img_name[data_idx-1], current_pose_T[0], current_pose_T[1], current_pose_T[2],
                                                                                                                    current_roll, current_pitch, current_yaw,
                                                                                                                    prev_pose_T[0], prev_pose_T[1], prev_pose_T[2], 
                                                                                                                    prev_roll, prev_pitch, prev_yaw]

                    self.dataset_writer.writerow(data)

                else:

                    print('Index 0 Skip')

        self.dataset_dict.close()

        self.dataset_dict = open('./' + self.name + '_dataset_dict.csv', 'r', encoding='utf-8')
        self.data_list = []
        self.reader = csv.reader(self.dataset_dict)
        next(self.reader)
        for row_data in self.reader:
            self.data_list.append(row_data)
            self.len += 1
        self.dataset_dict.close()

    def transform_func(self, current_img, prev_img, height=640, width=192):

        resize = transforms.Resize(size=(width, height))
        current_img = resize(current_img)
        prev_img = resize(prev_img)

        if random.random() >= 0.5:
            random_brightness = random.uniform(0.8, 1.2)
            current_img = TF.adjust_brightness(current_img, brightness_factor=random_brightness)
            prev_img = TF.adjust_brightness(prev_img, brightness_factor=random_brightness)

        if random.random() >= 0.5:
            random_contrast = random.uniform(0.8, 1.2)
            current_img = TF.adjust_contrast(current_img, contrast_factor=random_contrast)
            prev_img = TF.adjust_contrast(prev_img, contrast_factor=random_contrast)

        if random.random() >= 0.5:
            random_scale = random.uniform(1.0, 1.5)
            current_img = TF.affine(current_img, scale=random_scale, translate=(0, 0), angle=0, shear=0)
            prev_img = TF.affine(prev_img, scale=random_scale, translate=(0, 0), angle=0, shear=0)

        current_img = TF.to_tensor(current_img)
        prev_img = TF.to_tensor(prev_img)

        return current_img, prev_img

    # Dataset Load Function
    def __getitem__(self, index):
        
        current_data_row = self.data_list[index]

        current_img = Image.open(current_data_row[1]).convert('RGB')
        prev_img = Image.open(current_data_row[2]).convert('RGB')

        if self.verbose == 1:
            combined_img = Image.new('L', (prev_img.width + current_img.width, prev_img.height))
            combined_img.paste(prev_img, (0, 0))
            combined_img.paste(current_img, (prev_img.width, 0))
            plt.imshow(combined_img)
            plt.pause(0.001)
            plt.show(block=False)
            plt.clf()

        if self.mode == 'training':
            
            if self.train_transform is not None:

                # current_img = self.train_transform(current_img)
                # prev_img = self.train_transform(prev_img)
                current_img, prev_img = self.transform_func(current_img, prev_img)

        elif self.mode == 'validation':
            
            if self.valid_transform is not None:
                
                current_img = self.valid_transform(current_img)
                prev_img = self.valid_transform(prev_img)

        current_pose_x = float(current_data_row[3])
        current_pose_y = float(current_data_row[4])
        current_pose_z = float(current_data_row[5])
        current_pose_roll = float(current_data_row[6])
        current_pose_pitch = float(current_data_row[7])
        current_pose_yaw = float(current_data_row[8])

        prev_pose_x = float(current_data_row[9])
        prev_pose_y = float(current_data_row[10])
        prev_pose_z = float(current_data_row[11])
        prev_pose_roll = float(current_data_row[12])
        prev_pose_pitch = float(current_data_row[13])
        prev_pose_yaw = float(current_data_row[14])

        if self.cm is True:
            dx = 100 * (current_pose_x - prev_pose_x)
            dy = 100 * (current_pose_y - prev_pose_y)
            dz = 100 * (current_pose_z - prev_pose_z)

        else:
            dx = (current_pose_x - prev_pose_x)
            dy = (current_pose_y - prev_pose_y)
            dz = (current_pose_z - prev_pose_z)

        droll = current_pose_roll - prev_pose_roll
        dpitch = current_pose_pitch - prev_pose_pitch
        dyaw = current_pose_yaw - prev_pose_yaw

        if self.verbose == 1:
            print('--------------------------------------------')
            print(current_data_row)
            print('{}'.format(current_data_row[1]))
            print('{}'.format(current_data_row[2]))
            print('{} {} {} {} {} {}'.format(current_pose_x, current_pose_y, current_pose_z, current_pose_roll, current_pose_pitch, current_pose_yaw))
            print('{} {} {} {} {} {}'.format(prev_pose_x, prev_pose_y, prev_pose_z, prev_pose_roll, prev_pose_pitch, prev_pose_yaw))
            print('{} {} {} {} {} {}'.format(dx, dy, dz, droll, dpitch, dyaw))

        # Stack the image as indicated in DeepVO paper
        prev_current_stacked_img = np.asarray(np.concatenate([prev_img, current_img], axis=0))
        
        # Prepare 6 DOF pose vector between t-1 and t (dX dY dZ dRoll dPitch dYaw)
        #prev_current_odom = np.asarray([dx, dy, dz, droll, dpitch, dyaw])
        prev_current_odom = np.asarray([dx, dy, dz])
        
        return prev_current_stacked_img, prev_current_odom

    def __len__(self):

        return self.len
