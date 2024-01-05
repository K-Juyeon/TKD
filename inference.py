import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets
from torchvision.transforms import transforms
from sklearn.metrics import f1_score, precision_score, recall_score

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

def inference():
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    f1 = f1_score(all_targets, all_predictions, average='macro')  # macro 평균 사용
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')

    print("F1-score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)

if __name__ == '__main__':
    global device
    global testloader
    global criterion
    global model

    modelss = "vgg16bn" #vgg16bn
    batch_size = 128  # 1
    num_classes = 64

    test_data_path = 'F:\\tkd_data\\splitdata\\test'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_data_path, transform=transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda:0")

    if modelss == "vgg16bn":
        model = models.vgg16_bn(pretrained=False)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    else:
        model = ResNet50(num_classes=num_classes)

    checkpoint = torch.load('./checkpoint/VS.pt')
    model.load_state_dict(checkpoint['net'])
    # model.load_state_dict(torch.load('./checkpoint/RS.pt'))
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    inference()