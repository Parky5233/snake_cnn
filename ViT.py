import gc
import random

import fastai.vision.learner
import timm
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import PIL
from fastai.vision.all import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

# Note: this code does not run on Windows OS
'''
This code was written following various tutorials and examples:
- https://docs.fast.ai/tutorial.vision
- https://pypi.org/project/timm/#models
'''

#set_seed(42, True) #setting random seed for reproducability

'''
Aug 10
- Work on more hyperparameters (batch sizes) - in  prog
    * mod: reg, sam
    * bs: 32,64,128
    * e: 5, 15
'''

def scramble(unshuffled_path,train_size):
    '''
    This function scrambles test/train data after training. Keeping equal distribution and doing so randomly (seeded)
    Assumes folder "unshuffled_path" exists with folders of each class and their images
    Idea: use between training epochs or after training
    '''
    #looking at folder of snake data (classes only, not split by train/test)
    #os.chdir(unshuffled_path)
    if not os.path.isdir("shuffled_snake"):
        os.mkdir("shuffled_snake")
    for split in ["Train","Test"]:
        os.mkdir("shuffled_snake/"+split)
    #train images
    for species in os.listdir(unshuffled_path):
        if not os.path.isdir("shuffled_snake/Train/"+species):
            os.mkdir("shuffled_snake/Train/"+species)
        size = len(os.listdir(unshuffled_path+"/"+species))
        files = os.listdir(unshuffled_path+"/"+species)
        for img in range(int(size*train_size)):
            f = random.randint(0,len(files)-1)
            shutil.copyfile(unshuffled_path+"/"+species+"/"+files[f],"shuffled_snake/Train/"+species+"/"+files[f])
            files.pop(f)

    #test images
    for species in os.listdir(unshuffled_path):
        if not os.path.isdir("shuffled_snake/Test/"+species):
            os.mkdir("shuffled_snake/Test/"+species)
        for img in os.listdir(unshuffled_path+"/"+species):
            if img not in os.listdir("shuffled_snake/Train/"+species):
                shutil.copyfile(unshuffled_path+"/"+species+"/"+img, "shuffled_snake/Test/"+species+"/"+img)
    print("Files shuffled")

def set_seed(dls,x=42): #must have dls, as it has an internal random.Random
    random.seed(x)
    dls.rng.seed(x) #added this line
    np.random.seed(x)
    torch.manual_seed(x)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(x)

labels = os.listdir("unshuffled_snake") #class labeled derived from folder names

path = "snake_images"
#files = get_image_files(path) #getting all filenames for snake_images folder
batch_size = 16 #normally 32

#setting to 2nd GPU (note: if device has no 2nd GPU, error will raise)
torch.cuda.set_device(1)
torch.cuda.empty_cache()

dls = ImageDataLoaders.from_folder(path, train='train', valid='test', bs=batch_size, item_tfms=Resize(224), seed=42) #image dataloader


dls.device = default_device()
set_seed(dls)
#dls.rng.seed(42) #setting dataloader random seed for reproducability
#dls.show_batch()
#plt.show()
mod_name = 'vit_base_patch16_224' #specifying model type - ViT with patch size of 16 and image size 224
learn = fastai.vision.learner.vision_learner(dls, mod_name, metrics=error_rate)  # 'vit_base_patch16_224'

#learn.dls.rng.seed(42)
set_seed(learn.dls)
# learn = fastai.vision.learner.vision_learner(dls, 'vit_base_patch16_224_sam', metrics=error_rate)
# print(learn.lr_find())
# plt.show()
epochs = 5
lr = 1e-3
learn.fine_tune(epochs, lr)  # 0.001-0.002
learn.show_results()
print("epochs: " + str(epochs) + ", lr: " + str(lr))
#interp = Interpretation.from_learner(learn)
#interp.plot_top_losses(9, figsize=(15, 10))
#plt.show()

#presenting confusion matrix for initial training performance
#class_int = ClassificationInterpretation.from_learner(learn)
#class_int.plot_confusion_matrix()
#plt.show()
learn.export(str(mod_name) + str(epochs) + 'lr' + str(lr) + 'bs' +str(batch_size) +'.pkl') #exporting
shuffle = True
small_path = "small_snake_data"#"small_snake_data"

#fine-tuning model to small snake dataset
#learn.dls = ImageDataLoaders.from_folder("formatted_data_1", train='Train', valid='Test', bs=batch_size, item_tfms=Resize(224), seed=42) #loading in data for fine-tuning

def calc_cwb(beta=0.9):
    #calculates balance factor on effective number of samples
    #count number of images/class
    counts = [0]*len(os.listdir(small_path+"/Train"))
    i = 0
    tot = 0
    for folder in os.listdir(os.getcwd()+"/"+small_path+"/Train"):
        counts[i] = len(os.listdir(os.getcwd()+"/"+small_path+"/Train/"+folder))
        tot += counts[i]
        i += 1
    #calculate class-wise balance factor from ratios of classes
    cwb = [0]*len(counts)
    for index in range(len(counts)):
        cwb[index] = (1 - beta) / (beta ** counts[index])
    #returning weights as a tensor
    return torch.cuda.FloatTensor(cwb)

def calc_alph(beta=0.9):
    counts = [0] * len(os.listdir(small_path+"/Train"))
    i = 0
    tot = 0
    for folder in os.listdir(os.getcwd() + "/"+small_path+"/Train"):
        counts[i] = len(os.listdir(os.getcwd() + "/"+small_path+"/Train/" + folder))
        tot += counts[i]
        i += 1
    weights = []
    for count in counts:
        b = (count/tot)
        weights.append( (1 - b) / (len(counts)-1))
    return torch.cuda.FloatTensor(weights)

learn2 = load_learner(os.path.join(os.getcwd()+'/'+path,str(mod_name) + str(epochs) + 'lr' + str(lr) + 'bs' +str(batch_size) +'.pkl'))
#need to change output dimensions. Could look into nee way to load_state_dict
#Attempt to mod FocalLoss from:https://github.com/fastai/fastai/blob/master/fastai/losses.py#L104 for multi-class w/ imbalance
class MCFocalLoss(nn.Module):
    def __init__(self, gamma=2, weights = calc_cwb(),reduction='mean', logits=False):
        super(MCFocalLoss, self).__init__()
        self.gamma = gamma
        self.weights = weights
        self.logits = logits
        self.reduction = reduction

    #note loss has to be averaged over batch size
    def forward(self,inp:Tensor,targ:Tensor) -> Tensor:
        loss = 0
        #print(targ)
        for i in range(len(self.weights)):
            ce_loss = F.cross_entropy(inp, targ, weight=self.weights,reduction="none")#need to change calc_cwb to be dict w/ class names/targs
            pt = torch.exp(-ce_loss)
            temp = (1 - pt)**self.gamma * ce_loss
            if self.reduction == "mean":
                loss += temp.mean()
        return torch.cuda.FloatTensor(-1*loss)

class MCFocalReg(BaseLoss):
    def __init__(self, gamma:float=2, weights = calc_alph(),reduction:str='mean'):
        self.gamma = gamma
        self.weights = weights
        self.reduct = reduction

    #note loss has to be averaged over batch size
    def forward(self,inp:Tensor,targ:Tensor) -> Tensor:
        loss = 0
        print(targ)
        for i in range(len(self.weights)):
            ce_loss = F.cross_entropy(inp, targ, weight=self.weights[learn2.dls.vocab.index(targ)],reduction="none")#need to change calc_cwb to be dict w/ class names/targs
            pt = torch.exp(-ce_loss)
            temp = (1 - pt)**self.gamma * ce_loss
            if self.reduction == "mean":
                loss += temp.mean()
            elif self.reduction == "sum":
                loss += temp.sum()
        return torch.FloatTensor(loss)


epochs = 5
torch.cuda.set_device(1)
    #if epoch>1:
#learn2 = load_learner(os.path.join(os.getcwd()+'/'+small_path,str(mod_name) + str(epochs) + 'lr' + str(lr) + 'bs' +str(batch_size) +'.pkl'))#formatted_data_updated vs small_path
loss = CrossEntropyLossFlat(weight=calc_alph())#MCFocalReg()
dls = ImageDataLoaders.from_folder(small_path, train='Train', valid='Test', bs=batch_size, item_tfms=Resize(224), seed=42, loss_func=loss) #loading in data for fine-tuning #, loss_func=MCFocal
#setting to second gpu and setting random seed
learn2.dls = dls
#learn2.dl still contains the old classes from previous fine-tuning (learn)
#so confusion matrix returns error. Need to find a way to update
print(learn2)
print(learn2.dls.vocab)
dls.show_batch()
print(learn2.loss_func)
learn2.loss_func = loss
learn2.dls.device = default_device()
set_seed(learn2.dls)
#learn2.dls.rng.seed(42)
print("Model Loaded: "+str(os.path.join(os.getcwd()+'/'+path,str(mod_name) + str(epochs) + 'lr' + str(lr) + 'bs' +str(batch_size) +'.pkl')))
print("epochs: " + str(epochs) + ", lr: " + str(lr) +", bs: " +str(batch_size))
learn2.path = small_path
#print(learn2.model)
learn2.model[1][6] = nn.Linear(512,len(learn2.dls.vocab), bias=False)#changing classification layer to match new problem
#print(learn2.model)
learn2.fine_tune(epochs=epochs,base_lr=lr,freeze_epochs=1)#just running on frozen layer
if shuffle:
    small_path = "shuffled_snake"
    learn2.path = small_path
    loss = CrossEntropyLossFlat(weight=calc_alph())
    # loss = CrossEntropyLossFlat(weight=calc_alph())
    if not os.path.isdir("shuffled_snake"):
        scramble("unshuffled_snake", 0.85)
    dls = ImageDataLoaders.from_folder(small_path, train='Train', valid='Test', bs=batch_size,
                                           item_tfms=Resize(224),
                                           seed=42, loss_func=loss)  # loading in data for fine-tuning #, loss_func=MCFocal
    learn2.dls = dls
    #seems the final output layer is still size 12 instead of 11 for some reason?
    # learn2.eval()
    # print(learn2.all_batches())
class_int = ClassificationInterpretation.from_learner(learn2)#apparently fastai's confusion matrix breaks with a custom loss function
class_int.plot_confusion_matrix()
plt.savefig('cm_e'+str(epochs)+'_wce.png')

#learn2.fine_tune(secondary_epochs,lr) #fine-tuning model, freeze for 1 epoch then unfreeze and run epochs
#learn2.show_results()

#learn2.export(str(mod_name) + str(epochs) + 'lr' + str(lr) + 'bs' +str(batch_size) +'small_data.pkl') #saving fine-tuned model