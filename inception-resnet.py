from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os

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

datasets = []
channels = 3
dimx = 64
dimy = 64
height = round(sqrt(2876416/batch_size/16))
width = height
#loading all the data into a split of test and train
datasets.append(ImageFolder("train_data",transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])))#,transforms.Normalize(0.4678,0.2206)
datasets.append(ImageFolder("test_data",transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])))#,transforms.Normalize(0.4678,0.2206)

def disp_batch(data):
    for images, labels in data:
        fig,ax = plt.subplots(figsize = (16,12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
        break

from torch.utils.data.dataloader import DataLoader
#from torch.utils.data import random_split

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

'''
Implementing the Inception GoogLeNet arcitecture from https://arxiv.org/pdf/1409.4842.pdf

Need to make code work with aux_logits
'''
class inceptNet(nn.Module):
    '''
    Channels and parameters pulled from Table 1 of paper
    '''

    def __init__(self, aux_logits=True, classes=10):
        super(inceptNet, self).__init__()
        assert aux_logits == True or aux_logits == False
        self.aux_logits = aux_logits
        self.conv1 = convolutional_block(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = convolutional_block(64, 192, kernel_size=3, stride=1, padding=1)
        # use maxpool here again

        # order: in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, out_pool_1x1
        self.inception3a = inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception(256, 128, 128, 192, 32, 96, 64)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # use maxpool here again
        self.inception4a = inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception(512, 112, 114, 288, 32, 64, 64)
        self.inception4e = inception(528, 256, 160, 320, 32, 128, 128)
        # maxpool here
        self.inception5a = inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception(832, 384, 192, 384, 48, 128, 128)

        self.averagepool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, classes)

        #if self.aux_logits:
        #    self.aux1 = InceptAux(512, classes)
        #    self.aux2 = InceptAux(528, classes)
        #else:
        #    self.aux1 = self.aux2 = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.max_pool(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.max_pool2(x)

        x = self.inception4a(x)
        #if self.aux_logits and self.training:
        #    aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        #if self.aux_logits and self.training:
        #    aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.max_pool(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.averagepool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)
        #if self.aux_logits and self.training:
        #    return aux1, aux2, x
        #else:
        return x


class inception(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(inception, self).__init__()
        self.branch1 = convolutional_block(in_channels, out_1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            convolutional_block(in_channels, red_3x3, kernel_size=1),
            convolutional_block(red_3x3, out_3x3, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            convolutional_block(in_channels, red_5x5, kernel_size=1),
            convolutional_block(red_5x5, out_5x5, kernel_size=5, padding=2),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            convolutional_block(in_channels, out_1x1pool, kernel_size=1),
        )

    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1
        )


class InceptAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptAux, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = convolutional_block(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class convolutional_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(convolutional_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

my_model = inceptNet().to(device)
opt = torch.optim.Adam(my_model.parameters(), lr = learning_rate)
print("Optimizer = "+opt.__str__())
crit = nn.CrossEntropyLoss()
num_steps = len(train_loader)
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