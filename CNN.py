from math import sqrt

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
import os
#laptop code
#config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
#parameters
epoch_num = 15
batch_size = 64
learning_rate = 0.0001

os.chdir("snake_images")

species_classes = [fName for fName in os.listdir() if fName.endswith(".csv")]
for species in species_classes:
    species = species.split(".")[0]

'''
Idea: 
Grab images from all species
Shuffle data
Train with 85% of these images
Test with 15% of these images

Models:
- GoogLeNet(Inception-ResNet)
 * pretrained model that we refine with our data to refine to our task
 * simple transfer learning methods
- ResNet model <-- to start?
- Read Ch 14 and Ch 19
- in our gradudate studies we can do directed reading about ML to go through fundamentals of ML

-Train wheel project partnered with Research Ottawa
- With dropout I was able to achieve 22.95% accuracy. Not great, but a slight improvement.
The next step would be to improve the model structure and perhaps retry data norm. Update: reran and got 23.88%
'''

datasets = []
channels = 3
dimx = 64
dimy = 64
height = round(sqrt(2876416/batch_size/16))
width = height
#loading all the data into a split of test and train
datasets.append(ImageFolder("train_data",transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])))#,transforms.Normalize(0.4678,0.2206)
datasets.append(ImageFolder("test_data",transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])))#,transforms.Normalize(0.4678,0.2206)

def getStats(trainLoad):
    sum,squared_sum,num_batch = 0,0,0
    for data,_ in trainLoad:
        sum+= torch.mean(data,dim=[0,2,3])
        squared_sum += torch.mean(data**2,dim=[0,2,3])
        num_batch+=1
    mean = sum / num_batch
    std = (squared_sum/num_batch - mean**2)**0.5
    return mean,std

def disp_batch(data):
    for images, labels in data:
        fig,ax = plt.subplots(figsize = (16,12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
        break

#test data
test_size = len(datasets[1])
#train data
train_size = len(datasets[0])

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

train_data = datasets[0]
test_data = datasets[1]

print("Length of Train Data: ",len(train_data))
print("Length of Test Data: ",len(test_data))
print("Learning Rate = ",learning_rate)
print("Batch Size = ",batch_size)


train_loader = DataLoader(train_data,batch_size,shuffle=True,num_workers=0,pin_memory=True)
test_loader = DataLoader(test_data,batch_size*2,num_workers=0,pin_memory=True)

disp_batch(train_loader)
print(len(species_classes)," classes")
plt.show()
#displaying a batch of training data

'''


Last Updated: 5/3/2022 8:15am

Concerns to address:
- Low loss but also low accuracy. Perhaps CNN is not learning 
what I want it to. Could be overfitting.
- Batch normalization and dropout
- Model could be bottlenecked by data loading and accuracy calc since this is on the CPU
leading to low gpu util
- 30 epochs seems to be ideal
- laptop code uses anywhere between 15-25% gpu whereas PC uses 4% at most
could be a matter of quality of GPU/CPU
- normalizing appears to lower accuracy
- adam opt appears to be an improvement. Might need more data?

Understanding:
- how does the output size of convolutional layers impact performance/the model? How do we decide?
- how to know when to stop training cnn
- finding ideal batch size
- is having the same set of training/testing (where which ones we look at at a given time are randomized) ok?
- am i applied BN and dropout correctly?
Concerns: architecture could be flawed in terms of layers
overfitting could be an issue. 

Standard LeNet5
'''

class ConvNN(nn.Module):

    def __init__(self):
        super(ConvNN, self).__init__()
        #Cond2d(num colour channels,
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)# input = prev output channel size
        self.drop1 = nn.Dropout(p=0.2)
        self.full_con1 = nn.Linear(16*height*width, 120)
        self.drop2 = nn.Dropout(p=0.3)
        self.full_con2 = nn.Linear(120, 84)
        self.drop3 = nn.Dropout(p=0.5)
        self.full_con3 = nn.Linear(84, len(species_classes))

    def forward(self,x):
        x = self.drop1(self.pool(F.relu(self.conv1(x))))
        x = self.drop2(self.pool(F.relu(self.conv2(x))))
        x = x.view(-1, 16*height*width)#flattening tensor
        x = F.relu(self.full_con1(self.drop3(x)))
        x = F.relu(self.full_con2(x))
        x = self.full_con3(x)#no softmax needed here as we use CEL which includes it
        return x


#setting up model
my_model = ConvNN().to(device)
opt = torch.optim.Adam(my_model.parameters(), lr = learning_rate)

print("Optimizer = "+opt.__str__())
crit = nn.CrossEntropyLoss() #since we have multiple classes
num_steps = len(train_loader)
#trainloader size = (samplesize / batch num)
#training
my_model.train()
for epoch in range(epoch_num):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        #forward
        outputs = my_model(images)
        loss = crit(outputs,labels)

        #back and opt
        opt.zero_grad() #empty gradients
        loss.backward()
        opt.step()

        if(i+1)%50 == 0:
            print(f'Epoch [{epoch+1}/{epoch_num}], Step [{i+1}/{num_steps}], Loss: {loss.item():.4f}')

print("done training")
countDict = {}
my_model.eval()
#testing
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(len(species_classes))]
    n_class_samples = [0 for i in range(len(species_classes))]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = my_model(images)

        _,pred = torch.max(outputs,1)
        n_samples += labels.size(0)
        n_correct += (pred == labels).sum().item()
        #for some reason batch_size exceeds the number of labels
        for i in range(len(labels)): #could use batch_size*2 but, it's not always a perfect fit
            label = labels[i]
            #print(label)
            #print("Label = ",label)
            my_pred = pred[i]
            #print("Guess = ",my_pred)
            if (label == my_pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network {acc} %')
    for key in countDict.keys():
        print((str)()+" : "+(str)(countDict.get(key)))
    #division by zero is possible here? I'm assuming that means im overtraining for one species and we did not test that species
    #always guessing the same species. I don't think the data is properly shuffled
    for i in range(len(species_classes)):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {species_classes[i]}: {acc} %')
#graphical outputs
