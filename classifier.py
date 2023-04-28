import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

NUM_CLASSES = 10


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

         # dialete convolution
        
        self.conv1 = nn.Conv2d(1, 8, 3, dilation= 2)
        
        self.conv2 = nn.Conv2d(8, 16, 3, dilation= 3)                
        
        self.pool = nn.MaxPool2d(2, 2)

        self.bn1 = nn.BatchNorm2d(8)        
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm1d(24)
        self.bn4 = nn.BatchNorm1d(16)       
        
        self.fc1 = nn.Linear(16* 6 * 6, 24)
        self.fc2 = nn.Linear(24, 16)
        self.fc3 = nn.Linear(16, NUM_CLASSES)



    def forward(self, x):

        x = F.relu(self.conv1(x))                   
        x = self.pool(x)
        x = self.bn1(x)        
        
        x = F.relu(self.conv2(x))       

        x = self.bn2(x)

        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = self.bn3(x)   

        x = F.relu(self.fc2(x))
        x = self.bn4(x)   
        
        x = self.fc3(x)
        
        return x
