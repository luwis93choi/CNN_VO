from NN01_CNN_VO import CNN_VO
from NN02_CNN_VO_gray import CNN_VO_Gray
from NN04_CNN_GRU_VO import CNN_GRU

from dataloader import voDataLoader

from model_trainer import trainer
from model_tester import tester

from trainer_autoencoder import trainer_autoencoder
from tester_autoencoder import tester_autoencoder

from model_trainer_gray import trainer_gray
from model_tester_gray import tester_gray

from model_trainer_RNN import trainer_RNN
from model_tester_RNN import tester_RNN

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

train_sequence = ['00', '01', '02', '06', '10']
#train_sequence = ['01']
valid_sequence = ['04', '03', '05']
#test_sequence = ['07', '08', '09']
test_sequence = ['00']

normalize = transforms.Normalize(
    
    mean=[0.19007764876619865, 0.15170388157131237, 0.10659445665650864],
    std=[0.2610784009469139, 0.25729316928935814, 0.25163823815039915]
)

train_preprocess = transforms.Compose([
    transforms.Resize((192, 640)),
    transforms.CenterCrop((192, 640)),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0)], p=0.5),
    transforms.RandomApply([transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0)], p=0.5),
    transforms.RandomApply([transforms.RandomAffine(degrees=0, scale=(1, 1.2))], p=0.5),
    transforms.ToTensor(),
])

valid_preprocess = transforms.Compose([
    transforms.Resize((192, 640)),
    transforms.CenterCrop((192, 640)),
    transforms.ToTensor(),
])

if args['mode'] == 'train':

    if model_type == '1':

        print('CNN-based VO')

        NN_model = CNN_VO()

        model_trainer = trainer(NN_model=NN_model, use_cuda=True, cuda_num=cuda_num,
                                train_loader_preprocess_param=train_preprocess,
                                valid_loader_preprocess_param=valid_preprocess,
                                model_path=model_path,
                                img_dataset_path=img_dataset_path,
                                pose_dataset_path=pose_dataset_path,
                                learning_rate=learning_rate,
                                train_epoch=epoch, train_sequence=train_sequence, train_batch=batch_size,
                                valid_sequence=valid_sequence, valid_batch=1, 
                                plot_epoch=True,
                                sender_email=args['sender_email'], sender_email_pw=args['sender_pw'], receiver_email=args['receiver_email'])

    elif model_type == '2':

        print('CNN-based VO - Grayscale')

        NN_model = CNN_VO_Gray()

        model_trainer = trainer_gray(NN_model=NN_model, use_cuda=True, cuda_num=cuda_num,
                                    train_loader_preprocess_param=train_preprocess,
                                    valid_loader_preprocess_param=valid_preprocess,
                                    model_path=model_path,
                                    img_dataset_path=img_dataset_path,
                                    pose_dataset_path=pose_dataset_path,
                                    learning_rate=learning_rate,
                                    train_epoch=epoch, train_sequence=train_sequence, train_batch=batch_size,
                                    valid_sequence=valid_sequence, valid_batch=1, 
                                    plot_epoch=True,
                                    sender_email=args['sender_email'], sender_email_pw=args['sender_pw'], receiver_email=args['receiver_email'])

    elif model_type == '4':

        print('CNN-GRU based VO')

        NN_model = CNN_GRU()

        model_trainer = trainer_RNN(NN_model=NN_model, use_cuda=True, cuda_num=cuda_num,
                                    loader_preprocess_param=train_preprocess,
                                    model_path=model_path,
                                    img_dataset_path=img_dataset_path,
                                    pose_dataset_path=pose_dataset_path,
                                    learning_rate=learning_rate,
                                    train_epoch=epoch, train_sequence=train_sequence, train_batch=batch_size,
                                    valid_sequence=valid_sequence, valid_batch=batch_size, plot_epoch=True,
                                    sender_email=args['sender_email'], sender_email_pw=args['sender_pw'], receiver_email=args['receiver_email'])

    model_trainer.train()

elif args['mode'] == 'train_pretrained_model':

    if model_type == '1':

        print('CNN-based VO - Re-training')

        NN_model = CNN_VO()

        checkpoint = torch.load(model_path, map_location='cuda:')

        if checkpoint != None:
            print('Load complete')
        else:
            sys.exit('[main ERROR] Invalid checkpoint loading')

        model_trainer = trainer(NN_model=NN_model, checkpoint=checkpoint,
                                use_cuda=True, cuda_num=cuda_num,
                                loader_preprocess_param=train_preprocess,
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

        print('CNN-based VO - Test')

        NN_model = CNN_VO()

        if cuda_num is None:
            cuda_num = ''
            checkpoint = torch.load(model_path)
        
        else:
            checkpoint = torch.load(model_path, map_location='cuda:'+cuda_num)
            print('map_location : cuda:{}'.format(cuda_num))

        if checkpoint != None:
            print('Load complete')
        else:
            sys.exit('[main ERROR] Invalid checkpoint loading')

        model_tester = tester(NN_model=NN_model, checkpoint=checkpoint,
                            model_path=model_path,
                            use_cuda=True, cuda_num=cuda_num,
                            loader_preprocess_param=valid_preprocess,
                            img_dataset_path=img_dataset_path, 
                            pose_dataset_path=pose_dataset_path,
                            test_epoch=epoch, test_sequence=test_sequence, test_batch=batch_size,
                            plot_epoch=True,
                            sender_email=args['sender_email'], sender_email_pw=args['sender_pw'], receiver_email=args['receiver_email'])

    elif model_type == '2':

        print('CNN-based VO (Grayscale)- Test')

        NN_model = CNN_VO_Gray()

        if cuda_num is None:
            cuda_num = ''
            checkpoint = torch.load(model_path)
        
        else:
            checkpoint = torch.load(model_path, map_location='cuda:'+cuda_num)
            print('map_location : cuda:{}'.format(cuda_num))

        if checkpoint != None:
            print('Load complete')
        else:
            sys.exit('[main ERROR] Invalid checkpoint loading')

        model_tester = tester_gray(NN_model=NN_model, checkpoint=checkpoint,
                                model_path=model_path,
                                use_cuda=True, cuda_num=cuda_num,
                                loader_preprocess_param=valid_preprocess,
                                img_dataset_path=img_dataset_path, 
                                pose_dataset_path=pose_dataset_path,
                                test_epoch=epoch, test_sequence=test_sequence, test_batch=batch_size,
                                plot_epoch=True,
                                sender_email=args['sender_email'], sender_email_pw=args['sender_pw'], receiver_email=args['receiver_email'])

    elif model_type == '4':

        print('CNN-GRU based VO - Test')

        NN_model = CNN_GRU()

        checkpoint = torch.load(model_path, map_location='cuda:'+cuda_num)

        if checkpoint != None:
            print('Load complete')
        else:
            sys.exit('[main ERROR] Invalid checkpoint loading')

        model_tester = tester_RNN(NN_model=NN_model, checkpoint=checkpoint,
                                model_path=model_path,
                                use_cuda=True, cuda_num=cuda_num,
                                loader_preprocess_param=valid_preprocess,
                                img_dataset_path=img_dataset_path, 
                                pose_dataset_path=pose_dataset_path,
                                test_epoch=epoch, test_sequence=test_sequence, test_batch=batch_size,
                                plot_epoch=True,
                                sender_email=args['sender_email'], sender_email_pw=args['sender_pw'], receiver_email=args['receiver_email'])


    model_tester.run_test()
