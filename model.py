from torch import nn
import torch
import torchvision
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 5, padding=2),
                        nn.MaxPool2d(2),
                        nn.Flatten(),
                        nn.ReLU(),
                        nn.Linear(1024, 64),
                        nn.Linear(64, 10),
                    )

    def forward(self, x):
        output = self.model(x)
        return output 

class CXRModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CXRModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 64 * 64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        output = self.model(x)
        return output

class CXRModel2(nn.Module):
    def __init__(self, num_classes=11, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        old = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=False)

        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight.copy_(old.weight.sum(dim=1, keepdim=True))
        in_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_feat, num_classes)
        )

    def forward(self, x):
        output = self.backbone(x)
        return output

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    # torchvision.transforms.RandomHorizontalFlip(0.5),
    torchvision.transforms.RandomRotation(10),
    # torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
valid_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

after_t_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    torchvision.transforms.RandomRotation(10),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
