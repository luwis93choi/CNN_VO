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

class trainer():

    def __init__(self, NN_model=None,
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

        if NN_model == None:

            sys.exit('No NN model is specified')

        else:

            self.NN_model = NN_model
            self.NN_model.to(self.PROCESSOR)
            self.model_path = './'

        self.NN_model.train()
        self.NN_model.training = True

        self.train_loader_list = []
        for i in range(len(train_sequence)):
            self.train_loader_list.append(torch.utils.data.DataLoader(voDataLoader(img_dataset_path=self.img_dataset_path,
                                                                                   pose_dataset_path=self.pose_dataset_path,
                                                                                   transform=loader_preprocess_param,
                                                                                   sequence=train_sequence[i],
                                                                                   batch_size=self.train_batch),
                                                                                   batch_size=self.train_batch, shuffle=False, drop_last=True))

        self.translation_loss = nn.MSELoss()
        self.angular_loss = nn.MSELoss()
        self.loss = None

        self.optimizer = optim.Adam(self.NN_model.parameters(), lr=self.learning_rate, weight_decay=0.00001, amsgrad=True)

        summary(self.NN_model, Variable(torch.zeros((1, 6, 384, 1280)).to(self.PROCESSOR)))

        # Prepare Email Notifier
        self.notifier = notifier_Outlook(sender_email=self.sender_email, sender_email_pw=self.sender_pw)

    def train(self):

        start_time = str(datetime.datetime.now())

        training_loss = []

        for epoch in range(self.train_epoch):

            print('[EPOCH] : {}'.format(epoch))

            loss_sum = 0.0

            before_epoch = time.time()

            for train_loader in self.train_loader_list:

                print('-------')
                for batch_idx, (sequence, prev_current_img, prev_current_odom) in enumerate(train_loader):
                    
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

                        ### Model Train ###
                        estimated_T, estimated_R = self.NN_model(prev_current_img)

                        ### Loss Computation ###
                        loss_T = self.translation_loss(estimated_T.float(), prev_current_odom[:, :3].float())
                        loss_R = self.angular_loss(estimated_R.float(), prev_current_odom[:, 3:].float())

                        self.loss = loss_T + loss_R

                        ### Backpropagation & Parameter Update ###
                        self.optimizer.zero_grad()
                        self.loss.backward()
                        self.optimizer.step()

                        ### Accumulate total loss ###
                        loss_sum += float(self.loss.item())

                        print('[Epoch {}/{}][Sequence : {}][batch_idx : {}] - Batch Loss : {} / Total Loss : {}'.format(epoch, self.train_epoch ,sequence, batch_idx, float(self.loss.item()), loss_sum))

                        # print(prev_current_odom)
                        # print(estimated_T)
                        # print(estimated_R)

                        # print('loss T : {}'.format(loss_T.item()))
                        # print('loss R : {}'.format(loss_R.item()))

            after_epoch = time.time()

            training_loss.append(loss_sum)
            print(training_loss)

            #torch.save(self.NN_model, './CNN_VO_' + start_time + '.pth')
            torch.save({'epoch' : epoch,
                        'model_state_dict' : self.NN_model.state_dict(),
                        'optimizer_state_dict' : self.optimizer.state_dict(),
                        'loss' : self.loss}, './CNN_VO_state_dict_' + start_time + '.pth')

            if self.plot_epoch == True:
                plt.cla()
                plt.figure(figsize=(20,8))

                plt.xlabel('Training Length')
                plt.ylabel('Total Loss')
                plt.plot(range(len(training_loss)), training_loss, 'bo-')
                plt.title('CNN VO Training with KITTI [Total MSE Loss]\nTrain Sequence ' + str(self.train_sequence))
                plt.savefig(self.model_path + 'Training Results ' + start_time + '.png')