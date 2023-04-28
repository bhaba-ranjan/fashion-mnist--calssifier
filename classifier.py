import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

NUM_CLASSES = 10


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

      
        # fully connected NN
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 56)
        self.fc4 = nn.Linear(56, NUM_CLASSES)

        # convolutional NN
        # self.conv1 = nn.Conv2d(1, 64, 5)
        # self.conv2 = nn.Conv2d(64, 32, 3)
        # self.conv3 = nn.Conv2d(32, 16, 3)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(16 * 10 * 10, 120)
        # self.fc2 = nn.Linear(256, 266)
        # self.fc3 = nn.Linear(256, NUM_CLASSES)


      

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
