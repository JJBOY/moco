import torch.nn as nn


def LinCls(base_model, num_classes=1000, wrap=False):
    model = base_model(num_classes=num_classes)
    if wrap:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    return model
