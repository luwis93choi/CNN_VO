from dataloader_v2_gray import KITTI_Dataset_gray

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
from matplotlib import pyplot as plt
import sys
import os
import pickle

class trainer_gray_ensemble():

    def __init__(self, NN_model=None, checkpoint=None,
                       use_cuda=True, cuda_num='',
                       train_loader_preprocess_param=transforms.Compose([]), 
                       valid_loader_preprocess_param=transforms.Compose([]),
                       model_path='./',
                       img_dataset_path='', pose_dataset_path='',
                       learning_rate=0.001,
                       train_epoch=1, train_sequence=[], train_batch=1,
                       valid_sequence=[], valid_batch=1,
                       plot_epoch=True,
                       sender_email='', sender_email_pw='', receiver_email=''):

        self.use_cuda = use_cuda
        self.cuda_num = cuda_num

        self.img_dataset_path = img_dataset_path
        self.pose_dataset_path = pose_dataset_path
        self.model_path = model_path

        self.learning_rate = learning_rate

        self.train_loader_preprocess_param = train_loader_preprocess_param
        self.valid_loader_preprocess_param = valid_loader_preprocess_param

        self.train_epoch = train_epoch
        self.train_sequence = train_sequence
        self.train_batch = train_batch
        
        self.valid_epoch = train_epoch
        self.valid_sequence = valid_sequence
        self.valid_batch = valid_batch

        self.plot_epoch = plot_epoch

        self.sender_email = sender_email
        self.sender_pw = sender_email_pw
        self.receiver_email = receiver_email

        if (use_cuda == True) and (cuda_num != ''):        
            # Load main processing unit for neural network
            self.PROCESSOR = torch.device('cuda:'+self.cuda_num if torch.cuda.is_available() else 'cpu')

        else:
            self.PROCESSOR = torch.device('cpu')

        print(str(self.PROCESSOR))
        
        ### Model reloading ###
        if checkpoint != None:
            print('Re-training')

            if NN_model == None:

                sys.exit('[Trainer ERROR] No NN model is specified')

            else:

                self.NN_model = NN_model
                self.NN_model.load_state_dict(checkpoint['model_state_dict'])
                self.NN_model.to(self.PROCESSOR)

                # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # self.epoch = checkpoint['epoch']
                
                # self.train_loss = checkpoint['loss']
                
                self.model_path = './'

                print('Pre-trained model loaded')
        else: 

            if NN_model == None:

                sys.exit('No NN model is specified')

            else:

                self.NN_model = NN_model
                self.NN_model.to(self.PROCESSOR)
                self.model_path = './'

        Train_KITTI_Dataset = KITTI_Dataset_gray(name='KITTI_Train',
                                                                img_dataset_path=self.img_dataset_path,
                                                                pose_dataset_path=self.pose_dataset_path,
                                                                train_transform=train_loader_preprocess_param,
                                                                valid_transform=valid_loader_preprocess_param,
                                                                sequence=train_sequence, verbose=0)

        Valid_KITTI_Dataset = KITTI_Dataset_gray(name='KITTI_Valid',
                                                                img_dataset_path=self.img_dataset_path,
                                                                pose_dataset_path=self.pose_dataset_path,
                                                                train_transform=train_loader_preprocess_param,
                                                                valid_transform=valid_loader_preprocess_param,
                                                                sequence=valid_sequence, verbose=0)

        self.train_loader = torch.utils.data.DataLoader(Train_KITTI_Dataset, batch_size=self.train_batch, num_workers=8, shuffle=True, drop_last=True)

        self.valid_loader = torch.utils.data.DataLoader(Valid_KITTI_Dataset, batch_size=self.valid_batch, num_workers=8, shuffle=False, drop_last=True)

        self.optimizer = optim.Adam(self.NN_model.parameters(), lr=self.learning_rate, weight_decay=0.0001)

        self.train_translation_loss_x = nn.MSELoss()
        self.train_translation_loss_y = nn.MSELoss()
        self.train_translation_loss_z = nn.MSELoss()
        
        self.valid_translation_loss_x = nn.MSELoss()
        self.valid_translation_loss_y = nn.MSELoss()
        self.valid_translation_loss_z = nn.MSELoss()
        
        summary(self.NN_model, (torch.zeros((1, 6, 192, 640)).to(self.PROCESSOR)))

        self.NN_model.train()

        # Prepare Email Notifier
        self.notifier = notifier_Outlook(sender_email=self.sender_email, sender_email_pw=self.sender_pw)

    def train(self):

        start_time = str(datetime.datetime.now())

        training_loss = []
        valid_loss = []

        training_loss_x = []
        training_loss_y = []
        training_loss_z = []

        valid_loss_x = []
        valid_loss_y = []
        valid_loss_z = []

        plt.figure(figsize=(30,50))

        for epoch in range(self.train_epoch):
            
            print('----- [EPOCH] : {} -----'.format(epoch))

            train_loss_sum = 0.0
            train_T_loss_sum = 0.0
            train_R_loss_sum = 0.0
            
            train_loss_sum_x = 0.0
            train_loss_sum_y = 0.0
            train_loss_sum_z = 0.0

            train_length = 0

            valid_loss_sum = 0.0
            valid_T_loss_sum = 0.0
            valid_R_loss_sum = 0.0

            valid_loss_sum_x = 0.0
            valid_loss_sum_y = 0.0
            valid_loss_sum_z = 0.0

            valid_length = 0

            before_epoch = time.time()

            self.NN_model.train()
            for batch_idx, (prev_current_img, prev_current_odom) in enumerate(self.train_loader):

                ### Data GPU Transfer ###
                if self.use_cuda == True:
                    prev_current_img = prev_current_img.to(self.PROCESSOR)
                    prev_current_odom = prev_current_odom.to(self.PROCESSOR)

                ### Model Train ###
                estimated_x, estimated_y, estimated_z = self.NN_model(prev_current_img)

                # print('--------------------------------------')
                # print(estimated_pose_vect)
                # print(prev_current_odom)

                ### Backpropagation & Parameter Update ###
                self.optimizer.zero_grad()
                self.train_loss_x = self.train_translation_loss_x(estimated_x.float()[:, 0], prev_current_odom.float()[:, 0])
                self.train_loss_y = self.train_translation_loss_y(estimated_y.float()[:, 0], prev_current_odom.float()[:, 1])
                self.train_loss_z = self.train_translation_loss_z(estimated_z.float()[:, 0], prev_current_odom.float()[:, 2])
                
                # self.train_loss.backward()
                self.train_loss_x.backward(retain_graph=True)
                self.train_loss_y.backward(retain_graph=True)
                self.train_loss_z.backward()
                #torch.nn.utils.clip_grad_norm_(self.NN_model.parameters(), max_norm=1)
                self.optimizer.step()

                ### Translation/Rotation Loss ###
                T_loss_x = self.train_loss_x.item()
                T_loss_y = self.train_loss_y.item()
                T_loss_z = self.train_loss_z.item()

                ### Accumulate total loss ###
                train_loss_sum_x += T_loss_x
                train_loss_sum_y += T_loss_y
                train_loss_sum_z += T_loss_z
                train_length += 1

                updates = []
                updates.append('\n')
                updates.append('[Train Epoch {}/{}][Progress : {:.2%}][Batch Idx : {}] \n'.format(epoch, self.train_epoch, batch_idx/len(self.train_loader), batch_idx))
                updates.append('[Batch Loss] X Loss : {:.4f} / Y Loss : {:.4f} / Z Loss : {:.4f} \n'.format(T_loss_x, T_loss_y, T_loss_z))
                updates.append('[Average Loss] X Loss : {:.4f} / Y Loss : {:.4f} / Z Loss : {:.4f} \n'.format(train_loss_sum_x/train_length, train_loss_sum_y/train_length, train_loss_sum_z/train_length))
                final_updates = ''.join(updates)

                sys.stdout.write(final_updates)
                
                if batch_idx < len(self.train_loader)-1:
                    for line_num in range(len(updates)):
                        sys.stdout.write("\x1b[1A\x1b[2K")

            with torch.no_grad():

                self.NN_model.eval()
                for batch_idx, (prev_current_img, prev_current_odom) in enumerate(self.valid_loader):
                    
                    ### Data GPU Transfer ###
                    if self.use_cuda == True:
                        prev_current_img = prev_current_img.to(self.PROCESSOR)
                        prev_current_odom = prev_current_odom.to(self.PROCESSOR)

                    ### Model Train ###
                    estimated_x, estimated_y, estimated_z = self.NN_model(prev_current_img)

                    #self.valid_loss = self.valid_translation_loss(estimated_pose_vect.float(), prev_current_odom.float())
                    self.valid_loss_x = self.valid_translation_loss_x(estimated_x.float()[:, 0], prev_current_odom.float()[:, 0])
                    self.valid_loss_y = self.valid_translation_loss_y(estimated_y.float()[:, 0], prev_current_odom.float()[:, 1])
                    self.valid_loss_z = self.valid_translation_loss_z(estimated_z.float()[:, 0], prev_current_odom.float()[:, 2])
                    
                    ### Translation/Rotation Loss ###
                    # T_loss = self.valid_translation_loss(estimated_pose_vect.float(), prev_current_odom.float()).item()
                    # valid_T_loss_sum += T_loss
                    T_loss_x = self.valid_loss_x.item()
                    T_loss_y = self.valid_loss_y.item()
                    T_loss_z = self.valid_loss_z.item()

                    ### Accumulate total loss ###
                    # valid_loss_sum += float(self.valid_loss.item())
                    valid_loss_sum_x += T_loss_x
                    valid_loss_sum_y += T_loss_y
                    valid_loss_sum_z += T_loss_z
                    valid_length += 1

                    updates = []
                    updates.append('\n')
                    updates.append('[Valid Epoch {}/{}][Progress : {:.2%}][Batch Idx : {}] \n'.format(epoch, self.train_epoch, batch_idx/len(self.valid_loader), batch_idx))
                    # updates.append('Batch Loss : {:.4f} / Translation Loss : {:.4f} \n'.format(self.valid_loss.item(), T_loss))
                    updates.append('[Batch Loss] X Loss : {:.4f} / Y Loss : {:.4f} / Z Loss : {:.4f} \n'.format(T_loss_x, T_loss_y, T_loss_z))
                    # updates.append('Average Loss : {:.4f} / Avg Translation Loss : {:.4f} \n'.format(valid_loss_sum/valid_length, valid_T_loss_sum/valid_length))
                    updates.append('[Average Loss] X Loss : {:.4f} / Y Loss : {:.4f} / Z Loss : {:.4f} \n'.format(valid_loss_sum_x/train_length, valid_loss_sum_y/train_length, valid_loss_sum_z/train_length))
                    final_updates = ''.join(updates)

                    sys.stdout.write(final_updates)

                    if batch_idx < len(self.valid_loader)-1:
                        for line_num in range(len(updates)):
                            sys.stdout.write("\x1b[1A\x1b[2K")

            after_epoch = time.time()

            print('Epoch {} Complete | Time Taken : {:.2f} min'.format(epoch, (after_epoch-before_epoch)/60))

            self.NN_model.train()

            if epoch == 0:
                print('Creating save directory')
                os.mkdir('./' + start_time)

            training_loss_x.append(train_loss_sum_x/train_length)
            training_loss_y.append(train_loss_sum_y/train_length)
            training_loss_z.append(train_loss_sum_z/train_length)

            valid_loss_x.append(valid_loss_sum_x/valid_length)
            valid_loss_y.append(valid_loss_sum_y/valid_length)
            valid_loss_z.append(valid_loss_sum_z/valid_length)

            with open('./' + start_time + '/CNN VO Avg Training Loss X ' + start_time + '.txt', 'wb') as loss_file:
                pickle.dump(training_loss_x, loss_file)

            with open('./' + start_time + '/CNN VO Avg Training Loss Y ' + start_time + '.txt', 'wb') as loss_file:
                pickle.dump(training_loss_y, loss_file)

            with open('./' + start_time + '/CNN VO Avg Training Loss Z ' + start_time + '.txt', 'wb') as loss_file:
                pickle.dump(training_loss_z, loss_file)

            with open('./' + start_time + '/CNN VO Avg Validation Loss X ' + start_time + '.txt', 'wb') as loss_file:
                pickle.dump(valid_loss_x, loss_file)

            with open('./' + start_time + '/CNN VO Avg Validation Loss Y ' + start_time + '.txt', 'wb') as loss_file:
                pickle.dump(valid_loss_x, loss_file)

            with open('./' + start_time + '/CNN VO Avg Validation Loss Z ' + start_time + '.txt', 'wb') as loss_file:
                pickle.dump(valid_loss_x, loss_file)

            torch.save(self.NN_model, './' + start_time + '/CNN_VO_' + start_time + '.pth')
            torch.save({'epoch' : epoch,
                        'model_state_dict' : self.NN_model.state_dict(),
                        'optimizer_state_dict' : self.optimizer.state_dict(),
                        'loss_x' : self.train_loss_x,
                        'loss_y' : self.train_loss_y,
                        'loss_z' : self.train_loss_z}, './' + start_time + '/CNN_VO_state_dict_' + start_time + '.pth')

            if self.plot_epoch == True:
                plt.cla()
                plt.xlabel('Training Length')
                plt.ylabel('Total Loss')
                plt.plot(range(len(training_loss_x)), training_loss_x, 'bo-')
                plt.plot(range(len(training_loss_y)), training_loss_y, 'go-')
                plt.plot(range(len(training_loss_z)), training_loss_z, 'ro-')
                plt.plot(range(len(valid_loss_x)), valid_loss_x, 'bo--')
                plt.plot(range(len(valid_loss_y)), valid_loss_y, 'go--')
                plt.plot(range(len(valid_loss_z)), valid_loss_z, 'ro--')
                plt.title('CNN VO Training with KITTI [Average MSE Loss]\nTrain Sequence ' + str(self.train_sequence) + ' | Valid Sequence ' + str(self.valid_sequence) + '\nLearning Rate : ' + str(self.learning_rate) + ' Batch Size : ' + str(self.train_batch) + '\nPreprocessing : ' + str(self.train_loader_preprocess_param))
                plt.savefig('./' + start_time + '/Training Results ' + start_time + '.png')

                print('Epoch {} Complete | Model Saved \n'.format(epoch))
