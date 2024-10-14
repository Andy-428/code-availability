import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Define the path of the pre-trained weights
alexnet_pretrained_path = '/mnt/code/oiodnet/alexnet-owt-7be5be79.pth'
shufflenet_pretrained_path = '/mnt/code/oiodnet/shufflenetv2_x2_0-8be3c8ee.pth'

class OIODNet(nn.Module):
    def __init__(self, alexnet_pretrained_path=None, shufflenet_pretrained_path=None, pretrained=True):
        super(OIODNet, self).__init__()

        # load AlexNet
        model1 = models.alexnet()
        if pretrained and alexnet_pretrained_path:
            model1.load_state_dict(torch.load(alexnet_pretrained_path))
        self.features1 = nn.Sequential(
            model1.features,
            nn.Upsample(size=(7, 7), mode='bilinear', align_corners=False)  # 使用 bilinear 插值
        )

        # load ShuffleNetV2
        model2 = models.shufflenet_v2_x2_0()
        if pretrained and shufflenet_pretrained_path:
            model2.load_state_dict(torch.load(shufflenet_pretrained_path))
        self.features2 = nn.Sequential(
            model2.conv1,
            model2.maxpool,
            model2.stage2,
            model2.stage3,
            model2.stage4,
            model2.conv5,
            nn.Conv2d(2048, 256, 3, 1, 1),  # Adjust the number of output channels
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Adjust the number of channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.maxpool = nn.AdaptiveAvgPool2d(1)  # Global average pooling

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Dropout(p=0.3),  # Appropriately reduce the Dropout ratio
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 2)
        )

    def getcam(self, input, shape):
        w, h = shape
        feat = self.extract_features(input)
        weights = list(self.fc.parameters())[-2].unsqueeze(0)
        cam = feat.unsqueeze(1) * weights[:, :, :, None, None]
        cam = torch.sum(cam, dim=2)
        cam = F.interpolate(cam, size=(w, h), mode='bilinear', align_corners=False)
        return cam

    def extract_features(self, x):
        """
        Features from AlexNet and ShuffleNetV2 are extracted and spliced.
        """
        x1 = self.features1(x)
        x2 = self.features2(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.bottleneck(x)
        return x

    def forward(self, x):
        """
        Forward propagation process
        """
        x = self.extract_features(x)
        x = self.maxpool(x).view(x.size(0), -1)  # Global average pooling
        x = self.fc(x)
        return x