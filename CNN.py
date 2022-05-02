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

#need to ensure same amount of train and test of each species
#need to increase the number of species to 10
#increase number of samples within a species/class to 1000 or 1500
#batch numerization and dropout?

#config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
#parameters
epoch_num = 50
batch_size = 64
learning_rate = 0.01

os.chdir("snake_images")

species_classes = os.listdir()
print("Species:")
for species in species_classes:
    print(species)

os.chdir("..")
'''
Idea: 
Grab images from all species
Shuffle data
Train with 85% of these images
Test with 15% of these images
'''

#loading all the data into a split of test and train
dataset = ImageFolder("snake_images",transform = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()]))

def display_img(img,label):
    print("Label = ",dataset.classes[label])
    plt.imshow(img.permute(1,2,0))

def disp_batch(data):
    for images, labels in data:
        fig,ax = plt.subplots(figsize = (16,12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
        break

#display_img(*dataset[0])
#plt.show()
#showing first snake img
#test data
test_size = round(0.15 * len(dataset))
#train data
train_size = len(dataset) - test_size

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split


#randomly splitting the test and train data (I believe this effectively shuffles the data)
#need to ensure equal amounts of each class in both train or test
test_data,train_data = random_split(dataset,[test_size,train_size])

print("Length of Train Data: ",len(train_data))
print("Length of Test Data: ",len(test_data))

#placing the data into batches, pin_memory = true speed up allocation for space for training into GPU
train_loader = DataLoader(train_data,batch_size,shuffle=True,num_workers=0,pin_memory=True)
test_loader = DataLoader(test_data,batch_size*2,num_workers=0,pin_memory=True)

disp_batch(train_loader)
plt.show()
#displaying a batch of training data

'''
Concerns to address:
- Adjust layers for different sizes beyond 32x32 to increase accuracy
i.e. figuring out how to know the kernel size, padding, etc for each layer for 
any given size/dimension image
- could be early fitting. 
'''

#Conv Model (to be added to later)
class ConvNN(nn.Module):

    def __init__(self):
        super(ConvNN, self).__init__()
        #Cond2d(num colour channels,
        self.conv1 = nn.Conv2d(3,6,5)#last parameter must have # equal to number of classes
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)# input = prev output channel size
        self.full_con1 = nn.Linear(16*5*5, 120)
        self.full_con2 = nn.Linear(120, 84)
        self.full_con3 = nn.Linear(84, len(species_classes))

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)#flattening tensor
        x = F.relu(self.full_con1(x))
        x = F.relu(self.full_con2(x))
        x = self.full_con3(x)#no softmax needed here as we use CEL which includes it
        return x


#setting up model
my_model = ConvNN().to(device)
opt = torch.optim.SGD(my_model.parameters(), lr = learning_rate)
crit = nn.CrossEntropyLoss() #since we have multiple classes
num_steps = len(train_loader)
#trainloader size = (samplesize / batch num)
#training
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
        #print("Length of labels = ",len(labels))
        for i in range(len(labels)): #could use batch_size*2 but, it's not always a perfect fit
            label = labels[i]
            #print("Label = ",label)
            my_pred = pred[i]
            #print("Guess = ",my_pred)
            if (label == my_pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network {acc} %')
    #division by zero is possible here? I'm assuming that means im overtraining for one species and we did not test that species
    #always guessing the same species. I don't think the data is properly shuffled
    for i in range(len(species_classes)):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {species_classes[i]}: {acc} %')
#graphical outputs