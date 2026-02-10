# Файл: src/model.py
import torch
import torch.nn as nn
from torchvision import models

def get_pill_classifier(num_classes, freeze_backbone=True):
    """
    Создает модель на базе ResNet18 для классификации таблеток.
    """
    # Загружаем предобученную модель
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Замораживаем веса, если это необходимо
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Заменяем финальный слой (полносвязный классификатор)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    
    return model
