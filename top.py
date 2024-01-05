# -*- coding: utf-8 -*-

import torch
import torchvision.transforms as transforms
import os
from PIL import Image
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm

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

class_names = ['기본준비', '내려헤쳐막기', '돌려차고 앞굽이하고 아래막기', '돌려차고 앞굽이하고 얼굴바깥막고 지르기', '두발당성차고 앞굽이하고 안막고 두번지르기',
               '뒤꼬아서고 두주먹젖혀지르기', '뒤꼬아서고 등주먹앞치기', '뒷굽이하고 거들어바깥막기', '뒷굽이하고 거들어아래막기', '뒷굽이하고 바깥막기', '뒷굽이하고 손날거들어바깥막기',
               '뒷굽이하고 손날거들어아래막기', '뒷굽이하고 손날바깥막기', '뒷굽이하고 안막기', '뛰어앞차고 앞굽이하고 안막고 두번지르기', '모아서고 보주먹', '범서고 바탕손거들어안막고 등주먹앞치기',
               '범서고 바탕손안막기', '범서고 손날거들어바깥막기', '범서고 안막기', '앞굽이하고 가위막기', '앞굽이하고 거들어세워찌르기', '앞굽이하고 당겨지르기', '앞굽이하고 두번지르기', '앞굽이하고 등주먹앞치기',
               '앞굽이하고 등주먹앞치기하고 안막기', '앞굽이하고 바탕손안막고 지르기', '앞굽이하고 손날얼굴비틀어막기', '앞굽이하고 아래막고 안막기', '앞굽이하고 아래막고 지르기', '앞굽이하고 아래막기',
               '앞굽이하고 안막고 두번지르기', '앞굽이하고 안막기', '앞굽이하고 얼굴막기', '앞굽이하고 얼굴바깥막고 지르기', '앞굽이하고 얼굴지르기', '앞굽이하고 엇걸어아래막기', '앞굽이하고 외산틀막기',
               '앞굽이하고 제비품안치기', '앞굽이하고 지르기', '앞굽이하고 팔꿈치거들어돌려치기', '앞굽이하고 팔꿈치돌려치고 등주먹앞치기하고, 지르기', '앞굽이하고 팔꿈치표적치기', '앞굽이하고 헤쳐막기',
               '앞서고 등주먹바깥치기', '앞서고 손날안치기', '앞서고 아래막고 지르기', '앞서고 아래막기', '앞서고 안막고 지르기', '앞서고 안막기', '앞서고 얼굴막기', '앞서고 지르기', '앞차고 뒷굽이하고 바깥막기',
               '앞차고 범서고 바탕손안막기', '앞차고 앞굽이하고 등주먹앞치기', '앞차고 앞굽이하고 아래막고 안막기', '앞차고 앞굽이하고 지르기', '앞차고 앞서고 아래막고 지르기', '앞차고 앞서고 지르기',
               '옆서고 메주먹내려치기', '옆차고 뒷굽이하고 손날거들어바깥막기', '주춤서고 손날옆막기', '주춤서고 옆지르기', '주춤서고 팔꿈치표적치기']

top1_sucess = 0
top5_sucess = 0
filelist_sum = 0

model_path = 'C:\\TKD_VGG16\\checkpoint\\VA.pt'
# model = ResNet50(num_classes=64)
model = models.vgg16_bn(pretrained=False)  # If True, returns a model pre-trained on ImageNet
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 64)
model_dict = torch.load(model_path)
model.load_state_dict(model_dict['net'])
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

for root, dir, files in os.walk('F:\\tkd_data\\splitdata\\test') :
    filelist = os.listdir(root)
    filelist_jpg = [file for file in filelist if file.endswith(".jpg")]
    filelist_sum += len(filelist_jpg)
    for file in tqdm(filelist_jpg, desc='Processing') :
        image_path = os.path.join(root, file)
        label = image_path.split('\\')[4]

        image = Image.open(image_path)

        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        input_batch = input_batch.to(device)

        with torch.no_grad():
            input_batch = input_batch.to(device)
            outputs = model(input_batch)

        # Top-1 예측
        _, predicted_idx = torch.max(outputs, 1)
        top1_accuracy = class_names[predicted_idx.item()]

        # Top-5 예측
        _, top5_indices = torch.topk(outputs, 5)
        top5_accuracy = [class_names[idx.item()] for idx in top5_indices[0]]

        if label in top1_accuracy :
            top1_sucess += 1
        if label in top5_accuracy :
            top5_sucess += 1

        # 결과 출력
        # print(f"Top 1 정확도: {top1_accuracy}")
        # print(f"Top 5 정확도: {top5_accuracy}")
print(filelist_sum)
print(top1_sucess)
print(top5_sucess)