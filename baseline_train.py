import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.transforms import transforms
from torchvision import datasets

import argparse

import datetime
import time

import logging
import logging.handlers

from taekwon_dataset import DatasetFromFolder
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)
formatter = logging.Formatter('[%(asctime)s][%(levelname)sl%(filename)s:%(lineno)s >> %(message)s')

streamHandler = logging.StreamHandler()
fileHandler = logging.FileHandler('./baseline_train.log')

streamHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

logger.addHandler(streamHandler)
logger.addHandler(fileHandler)
logger.setLevel(level=logging.DEBUG)
writer = SummaryWriter()
    
def validate(epoch):
    
    global best_acc
    
    vgg16_ft.eval()
    valid_loss = 0
    correct = 0
    total = 0
    global early_stop


    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = vgg16_ft(inputs)
            loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #print(str(batch_idx))
            #print(str(len(testloader)))
            #print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  #% (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    writer.add_scalar('Loss/valid', valid_loss / total, epoch)
    writer.add_scalar('Accuracy/valid', (100. * correct / total), epoch)
            

        #print('Validation: Epoch=%d, Loss=%.3f, Acc=%.3f' % (epoch, valid_loss / total, 100. * correct / total))
       # print("--TestTime--%.2f----" % (time.time() - st))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        #print('Saving..')
        state = {
            'net': vgg16_ft.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/vgg16_64bs.t7')
        best_acc = acc
        print('Best Acc : %.3f' % best_acc)
        early_stop = 0
    else:
        early_stop = early_stop+1


def train(epoch):

    # train mode
    vgg16_ft.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_i, sample  in enumerate(trainloader):
        #print("inputs, targets : ", sample[0], sample[1])
        
        optimizer.zero_grad()
        
        #inputs, targets= sample[0], sample[1]
        inputs, targets= sample
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = vgg16_ft(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    writer.add_scalar('Loss/train', train_loss / total, epoch)
    writer.add_scalar('Accuracy/train', (100. * correct / total), epoch)
    
    print('Training   Epoch=%d, Loss=%.3f, Acc=%.3f' % (epoch, train_loss / total, 100. * correct / total))

if __name__ == '__main__':
    global device
    global trainloader
    global validloader
    global optimizer
    global criterion
    global vgg16_ft
    
    early_stop =0

    train_data_path = 'E:\\tkd_data\\splitdata\\train'
    test_data_path = 'E:\\tkd_data\\splitdata\\test'
    val_data_path = 'E:\\tkd_data\\splitdata\\validation'

    # 이미지 전처리를 위한 변환
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_data_path, transform=transform)
    # test_dataset = datasets.ImageFolder(test_data_path, transform=transform)
    val_dataset = datasets.ImageFolder(val_data_path, transform=transform)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    validloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=0)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_root', type=str, default='/tkd_data/3차 Split/1차_10', help='model.pt path(s)')
    # parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    #
    # opt = parser.parse_args()
    # print(opt)

    # data set & data loader
    # source = opt.data_root  #'/work/datatone_tkd/데이터톤 문제_sourcedata_preprocessing'
    print("Source dir : ", 'E:\\tkd_data\\splitdata')
    # dataset_train = DatasetFromFolder(dataset_dir=source,dataset_type='train')
    # trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=0)
    #
    # dataset_valid = DatasetFromFolder(dataset_dir=source,dataset_type='validation')
    # validloader = torch.utils.data.DataLoader(dataset_valid, batch_size=128, shuffle=True, num_workers=0)

    # device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    vgg16_ft = models.vgg16_bn(pretrained=False) #If True, returns a model pre-trained on ImageNet

    # vgg16_ft.features[0] = nn.Conv2d(9,64,kernel_size=(3,3),stride=1,padding=(1,1))
    num_ftrs = vgg16_ft.classifier[6].in_features
    vgg16_ft.classifier[6] = nn.Linear(num_ftrs,64)
    vgg16_ft = vgg16_ft.to(device)
    # print(vgg16_ft)

    # optimizer
    optimizer = optim.SGD(vgg16_ft.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # epoch times
    epoch_times = 100
    start_epoch = 0

    best_acc = 0
    start = time.time()
    
    for epoch in range(start_epoch, start_epoch + epoch_times):
        start_now = datetime.datetime.now()    
        print("lerning time (each epoch) Start : ", start_now)
        train(epoch)
        validate(epoch)
        end_now = datetime.datetime.now()
        print("lerning time (each epoch) End : ", end_now)
        # print("lerning time (each epoch) End : ", end_now)
        if early_stop > 5 :
            print("Early Stop")
            break;
    print("lerning time (100 epoch train+validate) : ", time.time() - start)