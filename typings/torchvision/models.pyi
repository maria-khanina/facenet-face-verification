from enum import Enum

from torch import nn

class ResNet18_Weights(Enum):
    IMAGENET1K_V1 = ...

class ResNet(nn.Module):
    fc: nn.Module

def resnet18(weights: ResNet18_Weights | None) -> ResNet: ...
