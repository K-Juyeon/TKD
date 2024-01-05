
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets
from torchvision.transforms import transforms
import time

import argparse
import os

from taekwon_dataset import DatasetFromFolder

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

def test():
    model.eval()
    valid_loss = 0
    correct = 0
    total = 0


    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            st = time.time()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        #print('Test: Loss=%.3f, Acc=%.3f' % (valid_loss / total, 100. * correct / total)) recall precision
        print('Test: Loss=%.3f, Acc=%.3f, Total=%d, Correct=%d' % (valid_loss / total, 100. * correct / total, total, correct))
        print("--TestTime--%.2f----" % (time.time() - st))

if __name__ == '__main__':
    global device
    global testloader
    global criterion
    global model

    modelss = "resnet50"
    batch_size = 128 # 1
    num_classes = 64

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_root', type=str, default='/work/datatone_tkd/데이터톤 문제_sourcedata_preprocessing', help='model.pt path(s)')
    # parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    #
    # opt = parser.parse_args()
    # print(opt)
    test_data_path = 'F:\\tkd_data\\splitdata\\test'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_data_path, transform=transform)

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # data set & data loader
    # source = opt.data_root  #'/work/datatone_tkd/데이터톤 문제_sourcedata_preprocessing'
    # dataset_test = DatasetFromFolder(dataset_dir=source,dataset_type='test')
    # testloader = torch.utils.data.DataLoader(dataset_test, batch_size=64,shuffle=False, num_workers=8)

    # device 
    device = torch.device("cuda:0")

    # model
    if modelss == "vgg16":
        model = models.vgg16_bn(pretrained=False)  # If True, returns a model pre-trained on ImageNet
        # model.features[0] = nn.Conv2d(9, 64, kernel_size=(3, 3), stride=1, padding=(1, 1))
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    else:
        model = ResNet50(num_classes=num_classes)

    # model
    # vgg16_ft = models.vgg16_bn(pretrained=False)
    # vgg16_ft.features[0] = nn.Conv2d(9,64,kernel_size=(3,3),stride=1,padding=(1,1))
    # num_ftrs = vgg16_ft.classifier[6].in_features
    # vgg16_ft.classifier[6] = nn.Linear(num_ftrs,64)
    # vgg16_ft = vgg16_ft.to(device)
    # print(vgg16_ft)

    # load train weights
    checkpoint = torch.load('./checkpoint/' + modelss + '_' + str(batch_size) + 'bs_' + 'sgd.pt')
    model.load_state_dict(checkpoint['net'])

    # criterion
    criterion = nn.CrossEntropyLoss()

    # test times
    test()
    
