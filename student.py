#!/usr/bin/env python3
"""
   student.py

   Submitted by Rushmila Islam (z5456038)

   UNSW ZZEN9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
a3main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os 

"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.

TBD
"""
############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    if mode == 'train':
        transform_train = transforms.Compose([
            transforms.Resize((224,224)),  
            transforms.RandomHorizontalFlip(p=0.7), 
            transforms.RandomRotation(45), 
            # transforms.RandomAutocontrast(p=0.2),
            # transforms.RandomAdjustSharpness(3, p=0.5),
            transforms.RandomVerticalFlip(p=0.3), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        # return transforms.ToTensor()
        return transform_train
    
    elif mode == 'test':
        transform_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(p=0.7), 
            transforms.RandomRotation(45), 
            # transforms.RandomAutocontrast(p=0.2),
            # transforms.RandomAdjustSharpness(3, p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        return transform_test
        # return transforms.ToTensor()


############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################

    
class Network(nn.Module):

    def __init__(self):
        super().__init__()

        # Convolution layers 
        #1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        #2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        #3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        #4
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        #5
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)


        # Fully connected layers 
        #6
        self.fc1 = nn.Linear(7*7*128, 1024)
        
        #7
        self.out = nn.Linear(1024, 8)

        # Adding dropouts to avoid overfitting
        self.dropout = nn.Dropout(0.5)
        

        
    def forward(self, x):
        #1
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        #2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        #3
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        #4
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        #5
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = F.adaptive_avg_pool2d(x, 7)

        x = torch.flatten(x, 1) # flatten all dimensions 

        #6
        x = F.relu(self.fc1(x))

        x = self.dropout(x)
        
        #7
        x = self.out(x)
        
        x = F.log_softmax(x, dim=-1)

        return x


net = Network()

file_path = 'savedModel.pth'

print('Network parameters: ',net)

    
############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
# optimizer = None 

optimizer = optim.Adam(net.parameters(), weight_decay=0.0001) 

loss_func = nn.CrossEntropyLoss() #CrossEntropyLoss for the multiclass classification


############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Normally, the default weight initialization and fixed learing rate
# should work fine. But, we have made it possible for you to define
# your own custom weight initialization and lr scheduler, if you wish.

def weights_init(m):
    if os.path.exists(file_path):
        net.load_state_dict(torch.load(file_path))
        print('loading from .pth')
        if isinstance(m, nn.Linear):
            print(m.weight)
            

    else:

        # For the Linear layer   
        print('inititalising ...')
        if isinstance(m, nn.Linear):
            print(m.weight)
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    
        # For the Convolution layer 
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    return 



# scheduler = None
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1) #, end_factor=0.5, total_iters=50

############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"

torch.manual_seed(7) #manual seed added

# train_val_split = 1 # train dataset
train_val_split = 1 # validation 

batch_size = 200 
epochs = 150

print('batch size: ', batch_size, 'epochs', epochs)
