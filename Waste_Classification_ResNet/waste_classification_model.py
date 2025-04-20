import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report




"""## CNN Model"""

# Define the CNN model with flexible input size
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Create a flatten layer to automatically handle dimensions
        self.flatten = nn.Flatten()
        # We'll set the fc1 input size dynamically after seeing the data
        self.fc1_size = None
        self.fc1 = None
        self.fc2 = nn.Linear(512, num_classes)
        
    def _setup_fc1(self, x):
        # Dynamically create fc1 based on the flatten output size
        c, h, w = x.shape[1], x.shape[2], x.shape[3]
        # Apply convolutions and pools to calculate output size
        h_out = h // 4  # Two MaxPool layers with kernel=2
        w_out = w // 4
        self.fc1_size = 64 * h_out * w_out
        self.fc1 = nn.Linear(self.fc1_size, 512).to(x.device)
        print(f"Dynamically created fc1 layer with input size: {self.fc1_size}, output size: 512")

    def forward(self, x):
        # Dynamically setup fc1 if it's the first forward pass
        if self.fc1 is None:
            self._setup_fc1(x)
            
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
""" complex cnn """
class ComplexCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # First convolution block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolution block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization
        
        # Create a flatten layer to automatically handle dimensions
        self.flatten = nn.Flatten()
        
        # We'll set the fc1 input size dynamically after seeing the data
        self.fc1_size = None
        self.fc1 = None
        
        # Add an additional fully connected layer (fc1), keeping the output size as 512
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def _setup_fc1(self, x):
        # Dynamically create fc1 based on the flatten output size
        c, h, w = x.shape[1], x.shape[2], x.shape[3]
        # Apply convolutions and pools to calculate output size
        h_out = h // 4  # Two MaxPool layers with kernel=2
        w_out = w // 4
        self.fc1_size = 64 * h_out * w_out
        self.fc1 = nn.Linear(self.fc1_size, 512).to(x.device)
        print(f"Dynamically created fc1 layer with input size: {self.fc1_size}, output size: 512")

    def forward(self, x):
        # Dynamically setup fc1 if it's the first forward pass
        if self.fc1 is None:
            self._setup_fc1(x)
            
        # Convolution layers with batch normalization
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Apply flattening
        x = self.flatten(x)
        
        # Fully connected layer (fc1)
        x = self.fc1(x)
        x = self.relu(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Final fully connected layer (fc2)
        x = self.fc2(x)
        
        return x


"""## CNN Model with ResNet Backbone for Transfer Learning"""
class ImprovedModel(nn.Module):
    def __init__(self, num_classes, use_pretrained=True):
        super().__init__()
        # Use ResNet18 as backbone for transfer learning
        self.backbone = models.resnet18(weights='DEFAULT' if use_pretrained else None)
        in_features = self.backbone.fc.in_features
        
        # Replace the final fully connected layer
        self.backbone.fc = nn.Identity()
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    
""" pre-trained ResNet with SEFR """
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)

        # Freeze all layers first
        for param in resnet.parameters():
            param.requires_grad = False

        # Unfreeze last block (layer4) and avgpool for fine-tuning
        for param in resnet.layer4.parameters():
            param.requires_grad = True
        for param in resnet.avgpool.parameters():
            param.requires_grad = True

        self.features = nn.Sequential(*list(resnet.children())[:-1])  # till avgpool

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x
