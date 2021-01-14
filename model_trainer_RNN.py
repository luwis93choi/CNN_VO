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
from matplotlib import pyplot as plt
import sys
import os
import pickle
import random

class trainer_RNN():

    def __init__(self, NN_model=None, checkpoint=None,
                       use_cuda=True, cuda_num='',
                       loader_preprocess_param=transforms.Compose([]), 
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

        self.loader_preprocess_param = loader_preprocess_param

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
                
                # self.loss = checkpoint['loss']
                
                self.model_path = './'

                print('Pre-trained model loaded')
        else: 

            if NN_model == None:

                sys.exit('No NN model is specified')

            else:

                self.NN_model = NN_model
                self.NN_model.to(self.PROCESSOR)
                self.model_path = './'


        self.train_loader_list = []
        for i in range(len(train_sequence)):
            self.train_loader_list.append(torch.utils.data.DataLoader(voDataLoader(img_dataset_path=self.img_dataset_path,
                                                                                pose_dataset_path=self.pose_dataset_path,
                                                                                transform=loader_preprocess_param,
                                                                                sequence=train_sequence[i],
                                                                                batch_size=self.train_batch),
                                                                                batch_size=self.train_batch, num_workers=16, shuffle=False, drop_last=True))
        
        self.valid_loader_list = []
        for i in range(len(valid_sequence)):
            self.valid_loader_list.append(torch.utils.data.DataLoader(voDataLoader(img_dataset_path=self.img_dataset_path,
                                                                                pose_dataset_path=self.pose_dataset_path,
                                                                                transform=loader_preprocess_param,
                                                                                sequence=valid_sequence[i],
                                                                                batch_size=self.valid_batch),
                                                                                batch_size=self.valid_batch, num_workers=16, shuffle=False, drop_last=True))

        self.optimizer = optim.Adam(self.NN_model.parameters(), lr=self.learning_rate, weight_decay=0.0001)

        self.total_train_length = 0
        for train_loader in self.train_loader_list:
            self.total_train_length += len(train_loader) 

        self.total_valid_length = 0
        for valid_loader in self.valid_loader_list:
            self.total_valid_length += len(valid_loader) 

        self.translation_loss = nn.MSELoss()
        self.rotation_loss = nn.MSELoss()
        
        self.hidden = (self.NN_model.init_hidden(self.train_batch)).to(self.PROCESSOR)
        summary(self.NN_model, (torch.zeros((self.train_batch, 6, 384, 1280)).to(self.PROCESSOR)), self.hidden)

        self.NN_model.train()

        # Prepare Email Notifier
        self.notifier = notifier_Outlook(sender_email=self.sender_email, sender_email_pw=self.sender_pw)

    def train(self):

        start_time = str(datetime.datetime.now())

        training_loss = []
        valid_loss = []

        terminal_cols, terminal_rows = os.get_terminal_size(0)

        for epoch in range(self.train_epoch):
            
            print('[EPOCH] : {}'.format(epoch))

            train_loss_sum = 0.0
            train_T_loss_sum = 0.0
            train_R_loss_sum = 0.0
            train_length = 0

            valid_loss_sum = 0.0
            valid_T_loss_sum = 0.0
            valid_R_loss_sum = 0.0
            valid_length = 0

            before_epoch = time.time()

            self.NN_model.train()

            # Shuffling training sequences
            random.shuffle(self.train_loader_list)
            
            for train_loader in self.train_loader_list:

                print('---------------------')
                data_idx = 0
                for batch_idx, (prev_current_img, prev_current_odom) in enumerate(train_loader):
                    
                    torch.autograd.set_detect_anomaly(True)

                    sequence = train_loader.dataset.sequence

                    if data_idx >= 1:

                        ### Data GPU Transfer ###
                        if self.use_cuda == True:
                            prev_current_img = prev_current_img.to(self.PROCESSOR)
                            prev_current_odom = prev_current_odom.to(self.PROCESSOR)

                        # print(prev_current_img.size())
                        # print(prev_current_odom.size())
                        
                        ### Model Train ###
                        estimated_pose_vect, self.hidden = self.NN_model(prev_current_img, self.hidden)
                        
                        # print('------------------')
                        # print(estimated_pose_vect.clone().detach())
                        # print(prev_current_odom.clone().detach())
                        
                        ### Backpropagation & Parameter Update ###
                        self.optimizer.zero_grad()
                        self.loss = self.translation_loss(estimated_pose_vect.float()[:, :, :3], prev_current_odom.float()[:, :, :3]) + 100 * self.rotation_loss(estimated_pose_vect.float()[:, :, 3:], prev_current_odom.float()[:, :, 3:])
                        self.loss.backward(retain_graph=True)
                        self.optimizer.step()

                        ### Translation/Rotation Loss ###
                        T_loss = self.translation_loss(estimated_pose_vect.float()[:, :, :3], prev_current_odom.float()[:, :, :3]).item()
                        train_T_loss_sum += T_loss

                        R_loss = 100 * self.rotation_loss(estimated_pose_vect.float()[:, :, 3:], prev_current_odom.float()[:, :, 3:]).item()
                        train_R_loss_sum += R_loss

                        ### Accumulate total loss ###
                        train_loss_sum += float(self.loss.item())
                        train_length += 1

                        updates = []
                        updates.append('\n')
                        updates.append('[Train Epoch {}/{}][Sequence : {}][Progress : {:.2%}][Batch Idx : {}] \n'.format(epoch, self.train_epoch ,sequence, batch_idx/len(train_loader), batch_idx))
                        updates.append('Batch Loss : {:.4f} / Translation Loss : {:.4f} / Rotation Loss : {:.4f} \n'.format(self.loss.item(), T_loss, R_loss))
                        updates.append('Average Loss : {:.4f} / Avg Translation Loss : {:.4f} / Avg Rotation Loss : {:.4f} \n'.format(train_loss_sum/train_length, train_T_loss_sum/train_length, train_R_loss_sum/train_length))
                        final_updates = ''.join(updates)

                        sys.stdout.write(final_updates)
                        
                        if batch_idx < len(train_loader)-1:
                            for line_num in range(len(updates)):
                                sys.stdout.write("\x1b[1A\x1b[2K")

                        data_idx += 1

                    else:
                        
                        print('Index 0, 1 Skip')

                        self.hidden = (self.NN_model.init_hidden(self.train_batch)).to(self.PROCESSOR)
                        print('GRU Hidden State Reset')

                        data_idx += 1

            with torch.no_grad():

                self.NN_model.eval()
                    
                for layer in self.NN_model.modules():
                    if isinstance(layer, nn.BatchNorm2d):
                        print('-----------------------------')
                        layer.track_running_stats = False
                        print('Disable {}'.format(layer))
                        print('layer.track_running_stats : {}'.format(layer.track_running_stats))
                
                for valid_loader in self.valid_loader_list:

                    print('---------------------')
                    data_idx = 0
                    for batch_idx, (prev_current_img, prev_current_odom) in enumerate(valid_loader):
                        
                        sequence = valid_loader.dataset.sequence

                        if data_idx >= 2:

                            ### Data GPU Transfer ###
                            if self.use_cuda == True:
                                prev_current_img = prev_current_img.to(self.PROCESSOR)
                                prev_current_odom = prev_current_odom.to(self.PROCESSOR)

                            ### Model Train ###
                            estimated_pose_vect, self.hidden = self.NN_model(prev_current_img, self.hidden)

                            self.loss = self.translation_loss(estimated_pose_vect.float()[:, :, :3], prev_current_odom.float()[:, :, :3]) + 100 * self.rotation_loss(estimated_pose_vect.float()[:, :, 3:], prev_current_odom.float()[:, :, 3:])
                        
                            ### Translation/Rotation Loss ###
                            T_loss = self.translation_loss(estimated_pose_vect.float()[:, :, :3], prev_current_odom.float()[:, :, :3]).item()
                            valid_T_loss_sum += T_loss

                            R_loss = 100 * self.rotation_loss(estimated_pose_vect.float()[:, :, 3:], prev_current_odom.float()[:, :, 3:]).item()
                            valid_R_loss_sum += T_loss
                            
                            ### Accumulate total loss ###
                            valid_loss_sum += float(self.loss.item())
                            valid_length += 1

                            updates = []
                            updates.append('\n')
                            updates.append('[Valid Epoch {}/{}][Sequence : {}][Progress : {:.2%}][Batch Idx : {}] \n'.format(epoch, self.train_epoch ,sequence, batch_idx/len(valid_loader), batch_idx))
                            updates.append('Batch Loss : {:.4f} / Translation Loss : {:.4f} / Rotation Loss : {:.4f} \n'.format(self.loss.item(), T_loss, R_loss))
                            updates.append('Average Loss : {:.4f} / Avg Translation Loss : {:.4f} / Avg Rotation Loss : {:.4f} \n'.format(valid_loss_sum/valid_length, valid_T_loss_sum/valid_length, valid_R_loss_sum/valid_length))
                            final_updates = ''.join(updates)

                            sys.stdout.write(final_updates)

                            if batch_idx < len(valid_loader)-1:
                                for line_num in range(len(updates)):
                                    sys.stdout.write("\x1b[1A\x1b[2K")
                                
                            data_idx += 1

                        else:

                            print('Index 0, 1 Skip')

                            self.hidden = (self.NN_model.init_hidden(self.train_batch)).to(self.PROCESSOR)
                            print('GRU Hidden State Reset')

                            data_idx += 1

            after_epoch = time.time()

            self.NN_model.train()

            if epoch == 0:
                print('Creating save directory')
                os.mkdir('./' + start_time)

            with open('./' + start_time + '/CNN_GRU VO Training Loss ' + start_time + '.txt', 'wb') as train_loss_file:
                pickle.dump(training_loss, train_loss_file)

            with open('./' + start_time + '/CNN_GRU VO Validation Loss ' + start_time + '.txt', 'wb') as valid_loss_file:
                pickle.dump(valid_loss, valid_loss_file)

            torch.save(self.NN_model, './' + start_time + '/CNN_GRU_VO_' + start_time + '.pth')
            torch.save({'epoch' : epoch,
                        'model_state_dict' : self.NN_model.state_dict(),
                        'optimizer_state_dict' : self.optimizer.state_dict(),
                        'loss' : self.loss}, './' + start_time + '/CNN_GRU_VO_state_dict_' + start_time + '.pth')

            training_loss.append(train_loss_sum/self.total_train_length)
            print('Epoch {} Complete | Time Taken : {:.2f} min'.format(epoch, (after_epoch-before_epoch)/60))
            print(training_loss)

            valid_loss.append(valid_loss_sum/self.total_valid_length)
            print('Epoch {} Complete | Time Taken : {:.2f} min'.format(epoch, (after_epoch-before_epoch)/60))
            print(valid_loss)

            if self.plot_epoch == True:
                plt.cla()
                plt.figure(figsize=(30,20))

                plt.xlabel('Training Length')
                plt.ylabel('Average Loss')
                plt.plot(range(len(training_loss)), training_loss, 'bo-')
                plt.plot(range(len(valid_loss)), valid_loss, 'ro-')
                plt.title('CNN-GRU VO Training with KITTI [Average MSE Loss]\nTrain Sequence ' + str(self.train_sequence) + '\nLearning Rate : ' + str(self.learning_rate) + ' Batch Size : ' + str(self.train_batch) + '\nPreprocessing : ' + str(self.loader_preprocess_param))
                plt.savefig('./' + start_time + '/Training vs Validation Results ' + start_time + '.png')

                plt.cla()
                plt.figure(figsize=(30,20))

                plt.xlabel('Training Length')
                plt.ylabel('Average Loss')
                plt.plot(range(len(training_loss)), training_loss, 'bo-')
                plt.title('CNN-GRU VO Training with KITTI [Average MSE Loss]\nTrain Sequence ' + str(self.train_sequence) + '\nLearning Rate : ' + str(self.learning_rate) + ' Batch Size : ' + str(self.train_batch) + '\nPreprocessing : ' + str(self.loader_preprocess_param))
                plt.savefig('./' + start_time + '/Training Loss-only Results ' + start_time + '.png')