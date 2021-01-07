from NN01_Branched_CNN import Branched_CNN

from dataloader import voDataLoader

from model_trainer import trainer
from model_tester import tester

import torch
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

from torchsummaryX import summary

import datetime
import numpy as np
from matplotlib import pyplot as plt

import argparse
import sys
import os

### Argument Parser
ap = argparse.ArgumentParser()

# NN-related argument
ap.add_argument('-m', '--mode', type=str, required=True, help='Setting the mode of neural network between training and test')
ap.add_argument('-n', '--type', type=str, required=True, help='Type of neural network to use')
ap.add_argument('-c', '--cuda_num', type=str, required=False, help='Specify which CUDA to use under multiple CUDA environment')
ap.add_argument('-s', '--model_path', type=str, required=True, help='Path for saving or loading NN model')
ap.add_argument('-i', '--img_dataset_path', type=str, required=True, help='Directory path to image dataset')
ap.add_argument('-p', '--pose_dataset_path', type=str, required=True, help='Directory path to pose dataset')
ap.add_argument('-e', '--epoch', type=int, required=True, help='Epoch for training and test')
ap.add_argument('-b', '--batch_size', type=int, required=True, help='Batch size for the model')
ap.add_argument('-l', '--learning_rate', type=float, required=True, help='Learning rate of the model')

# Notifier-related argument
ap.add_argument('-E', '--sender_email', type=str, required=False, help='Sender Email ID')
ap.add_argument('-P', '--sender_pw', type=str, required=False, help='Sender Email Password')
ap.add_argument('-R', '--receiver_email', type=str, required=False, help='Receiver Email ID')
args = vars(ap.parse_args())

model_type = args['type']

model_path = args['model_path']
img_dataset_path = args['img_dataset_path']
pose_dataset_path = args['pose_dataset_path']

cuda_num = args['cuda_num']
if cuda_num is None:
    cuda_num = ''

epoch = args['epoch']
batch_size = args['batch_size']
learning_rate = args['learning_rate']
train_sequence = ['00', '01', '02', '05', '06', '08', '09']
#train_sequence = ['01']
#train_sequence=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
test_sequence = ['01']

normalize = transforms.Normalize(
    
    # mean=[0.19007764876619865, 0.15170388157131237, 0.10659445665650864],
    # std=[0.2610784009469139, 0.25729316928935814, 0.25163823815039915]
    
    mean=[127. / 255., 127. / 255., 127. / 255.],
    std=[1 / 255., 1 / 255., 1 / 255.]
)

preprocess = transforms.Compose([
    transforms.Resize((384, 1280)),
    transforms.CenterCrop((384, 1280)),
    transforms.ToTensor(),
    normalize
])

# preprocess = transforms.Compose([
#     transforms.Resize((384, 1280)),
#     transforms.CenterCrop((384, 1280)),
#     transforms.ToTensor()
# ])

if args['mode'] == 'train':

    if model_type == '1':

        print('Branched_CNN')

        NN_model = Branched_CNN()

    model_trainer = trainer(NN_model=NN_model, use_cuda=True, cuda_num=cuda_num,
                            loader_preprocess_param=preprocess,
                            model_path=model_path,
                            img_dataset_path=img_dataset_path,
                            pose_dataset_path=pose_dataset_path,
                            learning_rate=learning_rate,
                            train_epoch=epoch, train_sequence=train_sequence, train_batch=batch_size,
                            plot_epoch=True,
                            sender_email=args['sender_email'], sender_email_pw=args['sender_pw'], receiver_email=args['receiver_email'])

    model_trainer.train()

elif args['mode'] == 'train_pretrained_model':

    if model_type == '1':

        print('Branched_CNN - Re-training')

        NN_model = Branched_CNN()

        checkpoint = torch.load(model_path)

        if checkpoint != None:
            print('Load complete')
        else:
            sys.exit('[main ERROR] Invalid checkpoint loading')

    model_trainer = trainer(NN_model=NN_model, checkpoint=checkpoint,
                             use_cuda=True, cuda_num=cuda_num,
                             loader_preprocess_param=preprocess,
                             model_path=model_path,
                             img_dataset_path=img_dataset_path,
                             pose_dataset_path=pose_dataset_path,
                             learning_rate=learning_rate,
                             train_epoch=epoch, train_sequence=train_sequence, train_batch=batch_size,
                             plot_epoch=True,
                             sender_email=args['sender_email'], sender_email_pw=args['sender_pw'], receiver_email=args['receiver_email'])

    model_trainer.train()


elif args['mode'] == 'test':

    if model_type == '1':

        print('Branched_CNN - Test')

        NN_model = Branched_CNN()

        checkpoint = torch.load(model_path, map_location='cuda:'+cuda_num)

        if checkpoint != None:
            print('Load complete')
        else:
            sys.exit('[main ERROR] Invalid checkpoint loading')

    model_tester = tester(NN_model=NN_model, checkpoint=checkpoint,
                           model_path=model_path,
                           use_cuda=True, cuda_num=cuda_num,
                           loader_preprocess_param=preprocess,
                           img_dataset_path=img_dataset_path, 
                           pose_dataset_path=pose_dataset_path,
                           test_epoch=epoch, test_sequence=test_sequence, test_batch=batch_size,
                           plot_epoch=True,
                           sender_email=args['sender_email'], sender_email_pw=args['sender_pw'], receiver_email=args['receiver_email'])

    model_tester.run_test()
