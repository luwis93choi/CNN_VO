from dataloader_for_autoencoder import voDataLoader

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

class trainer_autoencoder():

    def __init__(self, NN_model=None, checkpoint=None,
                       use_cuda=True, cuda_num='',
                       loader_preprocess_param=transforms.Compose([]), 
                       model_path='./',
                       img_dataset_path='', pose_dataset_path='',
                       learning_rate=0.001,
                       train_epoch=1, train_sequence=[], train_batch=1,
                       valid_sequence=[],
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
        self.valid_batch = train_batch

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

        self.pose_loss = nn.MSELoss(reduction='sum')
        self.autoencoder_loss = nn.MSELoss()

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


        self.optimizer = optim.Adam(self.NN_model.parameters(), lr=self.learning_rate)
        
        self.train_loader_list = []
        for i in range(len(train_sequence)):
            self.train_loader_list.append(torch.utils.data.DataLoader(voDataLoader(img_dataset_path=self.img_dataset_path,
                                                                                pose_dataset_path=self.pose_dataset_path,
                                                                                transform=loader_preprocess_param,
                                                                                sequence=train_sequence[i],
                                                                                batch_size=self.train_batch),
                                                                                batch_size=self.train_batch, num_workers=8, shuffle=True, drop_last=True))

        summary(self.NN_model, (torch.zeros((1, 9, 192, 640)).to(self.PROCESSOR)))

        self.NN_model.train()

        # Prepare Email Notifier
        self.notifier = notifier_Outlook(sender_email=self.sender_email, sender_email_pw=self.sender_pw)

    def io_img_disp(self, prev_current_img, recovered_prev_current_img, disp_fig):

        fig = disp_fig
            
        prev_img = np.transpose(prev_current_img.clone().detach().cpu().numpy()[0], (1, 2, 0))[:, :, :3]
        current_img = np.transpose(prev_current_img.clone().detach().cpu().numpy()[0], (1, 2, 0))[:, :, 3:]
        
        recovered_prev_img = ((np.transpose(recovered_prev_current_img.clone().detach().cpu().numpy()[0], (1, 2, 0))[:, :, :3])*255).astype(np.uint8)
        recovered_current_img = ((np.transpose(recovered_prev_current_img.clone().detach().cpu().numpy()[0], (1, 2, 0))[:, :, 3:])*255).astype(np.uint8)
        
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.title.set_text('Original Prev Image')
        plt.imshow(prev_img)

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.title.set_text('Original Current Image')
        plt.imshow(current_img)
        
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.title.set_text('Recovered Prev Image')
        plt.imshow(recovered_prev_img)

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.title.set_text('Recovered Current Image')
        plt.imshow(recovered_current_img)

        plt.pause(0.001)
        plt.show(block=False)
        plt.clf()

    def train(self):

        start_time = str(datetime.datetime.now())

        training_loss = []
        valid_loss = []

        io_img_fig = plt.figure(figsize=(10, 4))

        for epoch in range(self.train_epoch):

            estimated_x = 0.0
            estimated_y = 0.0
            estimated_z = 0.0

            GT_x = 0.0
            GT_y = 0.0
            GT_z = 0.0
            
            print('[EPOCH] : {}'.format(epoch))

            train_loss_sum = 0.0
            valid_loss_sum = 0.0

            before_epoch = time.time()

            self.NN_model.train()
            for train_loader in self.train_loader_list:

                print('-------')
                
                for batch_idx, (sequence, data_idx, prev_current_img, prev_current_odom) in enumerate(train_loader):
                    
                    if data_idx == 0:
                        print('Index 0 Skip')

                    else:

                        ### Data GPU Transfer ###
                        if self.use_cuda == True:
                            prev_current_img = prev_current_img.to(self.PROCESSOR)
                            prev_current_odom = prev_current_odom.to(self.PROCESSOR)

                        ### Model Train ###
                        recovered_prev_current_img, pose_est = self.NN_model(prev_current_img)

                        self.optimizer.zero_grad()
                        loss = self.autoencoder_loss(recovered_prev_current_img, prev_current_img) + self.pose_loss(pose_est.float(), prev_current_odom.float())
                        loss.backward()
                        self.optimizer.step()

                        ### Accumulate total loss ###
                        train_loss_sum += float(loss.item())

                        print('[Train Epoch {}/{}][Sequence : {}][Progress : {:.2%}][Batch Idx : {}] - Batch Loss : {:.4} / Total Loss : {:.4}'.format(epoch, self.train_epoch ,sequence, batch_idx/len(train_loader), batch_idx, loss.item(), train_loss_sum))

                        #self.io_img_disp(prev_current_img, recovered_prev_current_img, io_img_fig)
            '''
            self.NN_model.eval()
            with torch.no_grad():
                for train_loader in self.train_loader_list:

                    print('-------')
                    for batch_idx, (sequence, data_idx, prev_current_img, prev_current_odom) in enumerate(train_loader):
                        
                        if data_idx == 0:
                            print('Index 0 Skip')

                        else:

                            ### Data GPU Transfer ###
                            if self.use_cuda == True:
                                prev_current_img = prev_current_img.to(self.PROCESSOR)
                                prev_current_odom = prev_current_odom.to(self.PROCESSOR)

                            ### Model Train ###
                            recovered_prev_current_img, pose_est = self.NN_model(prev_current_img)

                            loss = self.autoencoder_loss(recovered_prev_current_img, prev_current_img) + self.pose_loss(pose_est.float(), prev_current_odom.float())
                        
                            ### Accumulate total loss ###
                            valid_loss_sum += float(loss.item())

                            print('[Valid Epoch {}/{}][Sequence : {}][Progress : {:.2%}][Batch Idx : {}] - Batch Loss : {:.4} / Total Loss : {:.4}'.format(epoch, self.train_epoch ,sequence, batch_idx/len(train_loader), batch_idx, loss.item(), valid_loss_sum))
                            
                            #self.io_img_disp(prev_current_img, recovered_prev_current_img, io_img_fig)
            '''

            after_epoch = time.time()

            if epoch == 0:
                print('Creating save directory')
                os.mkdir('./' + start_time)

            training_loss.append(train_loss_sum)
            #valid_loss.append(valid_loss_sum)
            print('Epoch {} Complete | Time Taken : {:.2f} min'.format(epoch, (after_epoch-before_epoch)/60))
            print(training_loss)

            with open('./' + start_time + '/CNN VO Training Loss ' + start_time + '.txt', 'wb') as loss_file:
                pickle.dump(training_loss, loss_file)

            torch.save(self.NN_model, './' + start_time + '/CNN_VO_' + start_time + '.pth')
            torch.save({'epoch' : epoch,
                        'model_state_dict' : self.NN_model.state_dict(),
                        'optimizer_state_dict' : self.optimizer.state_dict(),
                        'pose_loss' : self.pose_loss,
                        'autoencoder_loss' : self.autoencoder_loss}, './' + start_time + '/CNN_VO_state_dict_' + start_time + '.pth')

            if self.plot_epoch == True:
                plt.cla()
                plt.figure(figsize=(30,20))

                plt.xlabel('Training Length')
                plt.ylabel('Total Loss')
                plt.plot(range(len(training_loss)), training_loss, 'bo-')
                #plt.plot(range(len(valid_loss)), valid_loss, 'ro-')
                plt.title('Auto CNN VO Training with KITTI [Total MSE Loss]\nTrain Sequence ' + str(self.train_sequence) + '\nLearning Rate : ' + str(self.learning_rate) + ' Batch Size : ' + str(self.train_batch) + '\nPreprocessing : ' + str(self.loader_preprocess_param))
                plt.savefig('./' + start_time + '/Training Results ' + start_time + '.png')