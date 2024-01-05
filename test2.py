import os
import time
import datetime
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from taekwon_dataset import DatasetFromFolder
from torch.utils.tensorboard import SummaryWriter
# from torchinfo import summary
from torchsummary import summary
from torchvision import datasets
from torchvision.transforms import transforms

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# def model_info(model: nn.Module, batch: int, ch: int, width: int, hight: int, device: torch.device, depth=1):
#     model.eval()
#     col_names = ["input_size", "output_size", "num_params", "kernel_size", "mult_adds"]
#     img = torch.rand(batch, ch, width, hight).to(device)
#     summary(model, img.size(), None, None, col_names=col_names, depth=depth, verbose=1)


class ResNet50(nn.Module):
    def __init__(self, num_classes: int = 64):
        super(ResNet50, self).__init__()
        resnet50 = models.resnet50(pretrained=False)
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.max_pool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        self.gap = resnet50.avgpool
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# class vgg16_bn(nn.Module):
#     def __init__(self, num_classes: int = 64):
#         super(vgg16_bn, self).__init__()
#         vgg16bn = models.vgg16_bn(pretrained=False)
#         self.features = vgg16bn.features
#         self.avg_pool = nn.AdaptiveAvgPool2d(7)
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, num_classes)
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.features(x)
#         x = self.avg_pool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x


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


def validate(epoch, model):
    global best_acc

    model.eval()
    valid_loss = 0
    correct = 0
    total = 0
    global early_stop

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # print(str(batch_idx))
            # print(str(len(testloader)))
            # print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            # % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    writer.add_scalar('Loss/valid', valid_loss / total, epoch)
    writer.add_scalar('Accuracy/valid', (100. * correct / total), epoch)

    # print('Validation: Epoch=%d, Loss=%.3f, Acc=%.3f' % (epoch, valid_loss / total, 100. * correct / total))
    # print("--TestTime--%.2f----" % (time.time() - st))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, 'checkpoint/VS.pt')
        best_acc = acc
        print(f'Best Acc : {best_acc: .3f}')
        early_stop = 0
    else:
        early_stop = early_stop + 1


def train(epoch, model):
    # train mode
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    for batch_i, sample in enumerate(trainloader):
        # print("inputs, targets : ", sample[0], sample[1])

        optimizer.zero_grad()

        # inputs, targets= sample[0], sample[1]
        inputs, targets = sample
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
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
    global modelss

    modelss = "vgg16"  # vgg16 or resnet50
    early_stop = 5
    batch_size = 128
    num_classes = 64

    # 이미지 전처리를 위한 변환
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_data_path = 'F:\\PED\\splitdata\\train'

    val_data_path = 'F:\\PED\\splitdata\\val'

    train_dataset = datasets.ImageFolder(train_data_path, transform=transform)
    val_dataset = datasets.ImageFolder(val_data_path, transform=transform)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda:0")

    # model
    if modelss == "vgg16":
        model = models.vgg16_bn(pretrained=False)  # If True, returns a model pre-trained on ImageNet
        # model.features[0] = nn.Conv2d(9, 64, kernel_size=(3, 3), stride=1, padding=(1, 1))
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    else:
        model = ResNet50(num_classes=num_classes)

    model = model.to(device)
    # model_info(model, 1, 3, 224, 224, device, 4)
    # print(summary(model, input_size=(3, 224, 224)))

    # optimizer
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # criterion
    criterion = nn.CrossEntropyLoss()

    # epoch times
    epoch_times = 100
    start_epoch = 0

    best_acc = 0
    start = time.time()

    for epoch in range(start_epoch, start_epoch + epoch_times):
        start_now = datetime.datetime.now()
        print(f"learning time (each epoch) Start : ", start_now)
        train(epoch, model)
        validate(epoch, model)
        end_now = datetime.datetime.now()
        print(f"learning time (each epoch) End : ", end_now)
        if early_stop > 5:
            print("Early Stop")
            break
    print(f"learning time (100 epoch train+validate) : ", time.time() - start)