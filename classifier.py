import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

NUM_CLASSES = 10


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

      
        # fully connected NN
#         self.fc1 = nn.Linear(28 * 28, 256)
#         self.fc2 = nn.Linear(256, NUM_CLASSES)

        # convolutional NN set criteria
        
#         self.conv1 = nn.Conv2d(1, 8, 5)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.bn1 = nn.BatchNorm2d(8)
        
#         self.conv2 = nn.Conv2d(8, 16, 3)                
#         self.bn2 = nn.BatchNorm2d(16)
        
#         self.conv3 = nn.Conv2d(16, 32, 1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.bn3 = nn.BatchNorm2d(32)                
        
#         self.fc1 = nn.Linear(32 * 5 * 5, 24)
#         self.fc2 = nn.Linear(24, 16)
#         self.fc3 = nn.Linear(16, NUM_CLASSES)


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
#         B = x.shape[0]
#         x = x.view(B, -1)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)

#         x = F.relu(self.conv1(x))  
#         x = F.relu(self.convD(x)) 
# #         print("after c1",x.shape)        
#         x = self.pool1(x) 
#         x = self.bn1(x)
        
#         x = F.relu(self.conv2(x))
# #         print("after c2",x.shape)
        
#         x = self.bn2(x)
#         x = F.relu(self.conv3(x))        
# #         print("after c3",x.shape)
        
# #         x = self.pool(x)
# #         print("after pool",x.shape)
#         x = self.bn3(x)
    
#         x = x.view(x.shape[0], -1)
        
#         x = F.relu(self.fc1(x))
# #         print('relu:1 applied')

#         x = F.relu(self.fc2(x))
    
#         x = self.fc3(x)



        x = F.relu(self.conv1(x))                   
        x = self.pool(x)
        x = self.bn1(x)
        # print("after c1 ",x.shape)     
        # print("after c1 and pooling",x.shape)     
        
        
        x = F.relu(self.conv2(x))
        # print("after c2",x.shape)        

        x = self.bn2(x)

#         x = F.relu(self.conv3(x))     
#         x = self.bn3(x)   

#         x = F.relu(self.conv4(x))             
#         # print("after c3",x.shape)        

#         x = self.bn4(x)   
# #         print(x.shape)
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = self.bn3(x)   
#         print('relu:1 applied')

        x = F.relu(self.fc2(x))
        x = self.bn4(x)   
        
        x = self.fc3(x)
        
        return x
