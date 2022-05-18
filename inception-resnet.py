import time
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
print("Cuda Available: ",torch.cuda.is_available())

#parameters
epoch_num = 15
batch_size = 64
learning_rate = 0.001

os.chdir("snake_images")


species_classes = [fName for fName in os.listdir() if fName.endswith(".csv")]
for species in species_classes:
    species = species.split(".")[0]

datasets = []
channels = 3\
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

Need to make code work with auxiliary classifiers
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

        if self.aux_logits:
            self.aux1 = InceptAux(512, classes)
            self.aux2 = InceptAux(528, classes)
        else:
            self.aux1 = self.aux2 = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.max_pool(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.max_pool2(x)

        x = self.inception4a(x)
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.max_pool(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.averagepool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)
        if self.aux_logits and self.training:
            return aux1, aux2, x
        else:
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
        #self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.conv(x))#self.batchnorm()
since = time.time()
my_model = inceptNet().to(device)
opt = torch.optim.SGD(my_model.parameters(),lr=learning_rate,momentum=0.09)
print("Optimizer = "+opt.__str__())

file = "epoch_"+str(epoch_num)+"_batch_"+str(batch_size)+"_lr_"+str(learning_rate)+"_opt_adam_aux_true.txt"
crit = nn.CrossEntropyLoss()
num_steps = len(train_loader)
my_model.train()
for epoch in range(epoch_num):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        #forward
        #this model assumes auxiliary classifiers are being used
        auxout1, auxout2, outputs = my_model(images)
        loss0 = crit(outputs,labels)
        loss1 = crit(auxout1,labels) * 0.3
        loss2 = crit(auxout2, labels) * 0.3
        loss = loss0+loss1+loss2
        #back and opt
        opt.zero_grad() #empty gradients
        loss.backward()
        opt.step()

        if(i+1)%50 == 0:
            print(f'Epoch [{epoch+1}/{epoch_num}], Step [{i+1}/{num_steps}], Loss: {loss.item():.4f}')

print("done training")
time_tot = time.time() - since
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
        for i in range(len(labels)):
            label = labels[i]
            my_pred = pred[i]
            if (label == my_pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
    acc = 100.0 * n_correct / n_samples
    with open('../outputs/'+file,'w') as f:
        f.write("Length of Train Data: "+str(len(train_data))+"\n")
        f.write("Length of Test Data: "+str(len(test_data))+"\n")
        f.write("Learning Rate: "+str(learning_rate)+"\n")
        f.write("Batch Size: "+str(batch_size)+"\n")
        f.write(str(len(species_classes))+" Classes\n")
        f.write(opt.__str__()+"\n")
        print(f'Accuracy of the network {acc} %')
        f.write("Accuracy of Network: "+str(acc)+"%\n")
        #for key in countDict.keys():
        #    print((str)()+" : "+(str)(countDict.get(key)))
        for i in range(len(species_classes)):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {species_classes[i]}: {acc} %')
            f.write("Accuracy of "+species_classes[i]+": "+str(acc)+"%\n")
        print("Training complete in {:.0f}m {:.0f}s".format(time_tot // 60, time_tot % 60))
        f.write("Time: ",time_tot/60,"m ",time_tot%60,"s")