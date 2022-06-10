import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import os

'''
Using the pretrained Pytorch Inception-V3 and finetuning it to our needs
https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
'''

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
#print("Cuda Available: ",torch.cuda.is_available())

os.chdir("snake_images")
species_classes = [fName for fName in os.listdir() if fName.endswith(".csv")]
for species in species_classes:
    species = species.split(".")[0]
#parameters
epoch_num = 15
batch_size = 64
learning_rate = 0.001
class_num = len(species_classes)

#parameters for automation
epoch_set = [1,2,15,30,50]
batch_set = [32,64,128]
learn_rates = [0.01,0.001,0.0001]

for epochs in epoch_set:
    for batch in batch_set:
        for lr in learn_rates:
            datasets = []
            #loading all the data into a split of val and train
            datasets.append(ImageFolder("train_data",transform = transforms.Compose([transforms.Resize((299,299)),transforms.ToTensor()])))#,transforms.Normalize(0.4678,0.2206)
            datasets.append(ImageFolder("test_data",transform = transforms.Compose([transforms.Resize((299,299)),transforms.ToTensor()])))#,transforms.Normalize(0.4678,0.2206)

            def disp_batch(data):
                for images, labels in data:
                    fig,ax = plt.subplots(figsize = (16,12))
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
                    break

            from torch.utils.data.dataloader import DataLoader

            train_data = datasets[0]
            test_data = datasets[1]

            print("Length of Train Data: ",len(train_data))
            print("Length of Test Data: ",len(test_data))
            print("Learning Rate = ",lr)
            print("Batch Size = ",batch)
            #still need to work on finetuning this model
            train_loader = DataLoader(train_data,batch,shuffle=True,num_workers=0,pin_memory=True)
            val_loader = DataLoader(test_data,batch*2,num_workers=0,pin_memory=True)
            dataloaders = {'train':train_loader,'val':val_loader}
            #disp_batch(train_loader)
            print(len(species_classes)," classes")
            plt.show()

            model = models.inception_v3(pretrained=True).to(device)
            num_features = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_features,class_num)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features,class_num)

            data_transforms = {
                'train': transforms.Compose([
                    transforms.RandomResizedCrop(299),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'val': transforms.Compose([
                    transforms.Resize(299),
                    transforms.CenterCrop(299),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            }

            #Parameters to optimize
            params_to_update = model.parameters()
            '''
            print("Params to learn:")
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)
            '''
            #optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)
            optimizer = optim.Adam(params_to_update,lr=lr)
            crit = nn.CrossEntropyLoss()

            #Training and Eval
            since = time.time()
            val_history = []
            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc = 0.0
            model = model.to(device)
            #print(model)
            n_samples = 0
            n_correct = 0
            n_class_correct = [0 for i in range(len(species_classes))]
            n_class_samples = [0 for i in range(len(species_classes))]
            confusionMat = np.zeros((len(species_classes),len(species_classes)))
            for epoch in range(epochs):
                print('Epoch {}/{}'.format(epoch,epochs-1))
                print('-'*10)

                for phase in ['train','val']:
                    if phase == 'train':
                        model.train()
                    else:
                        model.eval()

                    running_loss = 0.0
                    running_corrects = 0

                    #Iterating Data
                    for inputs, labels in dataloaders[phase]:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        optimizer.zero_grad()

                        #forward
                        with torch.set_grad_enabled(phase == 'train'):
                            if phase == 'train':
                                outputs, aux_ouputs = model(inputs)
                                loss1 = crit(outputs,labels)
                                loss2 = crit(aux_ouputs,labels)
                                loss = loss1 + loss2*0.4
                            else:
                                outputs = model(inputs)
                                loss = crit(outputs,labels)

                            _,preds = torch.max(outputs,1)

                            if phase == 'val':
                                n_samples += labels.size(0)
                                n_correct += (preds == labels).sum().item()
                                for i in range(len(labels)):
                                    label = labels[i]
                                    my_pred = preds[i]
                                    if confusionMat[label][my_pred] == None:
                                        confusionMat[label][my_pred] = 1
                                    else:
                                        confusionMat[label][my_pred] += 1
                                    if(label == my_pred):
                                        n_class_correct[label] += 1
                                    n_class_samples[label] += 1
                            #back
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                    epoch_loss = running_loss / len(dataloaders[phase].dataset)
                    epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                    # deep copy the model
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                    if phase == 'val':
                        val_history.append(epoch_acc)

                print()

            time_tot = time.time() - since
            file = str(epochs)+"_e_"+str(batch)+"_b_"+str(lr)+"_lr.txt"
            with open('../outputs/' + file, 'w') as f:
                f.write("Runtime: "+str(time_tot//60)+"m "+str(time_tot%60)+"s\n")
                f.write("Best Val Acc: "+str(best_acc)+"\n")
                f.write("Overall Acc: "+str(100.0*n_correct/n_samples)+"\n")
                for i in range(len(species_classes)):
                    acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                    f.write("Accuracy of "+str(species_classes[i])+": "+str(acc)+"\n")
                for i in range(len(species_classes)):
                    for j in range(len(species_classes)):
                        f.write(str(confusionMat[i][j])+" ")
                    f.write("\n")
                for i in range(len(species_classes)):
                    f.write(str(i)+" = "+str(species_classes[i]))
            print("Training complete in {:.0f}m {:.0f}s".format(time_tot // 60, time_tot % 60))
            print("Best val acc: {:4f}".format(best_acc))
            print("Overall Acc: "+str(100.0*n_correct/n_samples))
            for i in range(len(species_classes)):
                acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                print(f'Accuracy of {species_classes[i]}: {acc} %')
            print("\n")
            #printing confusion matrix
            for i in range(len(species_classes)):
                for j in range(len(species_classes)):
                    print(str(confusionMat[i][j])+" ", end='')
                print(" ")
            for i in range(len(species_classes)):
                print(str(i)+" = "+species_classes[i]+"\n")
#load model weights
model.load_state_dict(best_model_wts)
# plt.title("Accuracy vs. Number of Epochs")
# plt.xlabel("Training Epochs")
# plt.ylabel("Validation Accuracy")
# plt.plot(range(1,epoch_num+1),val_history,label="Pretrained")
# plt.ylim((0,1.))
# plt.xticks(np.arange(1,epoch_num+1,1.0))
# plt.legend()
# plt.savefig('scratch-net.pdf')
# plt.show()