import gc
import fastai.vision.learner
import timm
import os
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

This code implements a ViT model for a binary classification task
on defective vs normal train images. Uses FocalLossFlat to handle class imbalance
and a 85/15 split on train/test data.

Note: need to check that weighting function is working correctly
'''

def calc_cwb():
    #calculates balance factor on effective number of samples (smoothed)
    #count number of images/class
    # counts = [0]*len(learn2.dls.vocab)
    # i = 0
    # tot = 0
    # for folder in os.listdir(os.getcwd()+"/formatted_data_updated/Train"):
    #     counts[i] = len(os.listdir(os.getcwd()+"/formatted_data_updated/Train/"+folder))
    #     tot += counts[i]
    #     i += 1
    counts = [2000, 1000]
    tot = 3000
    v_tot = 0
    weights = []
    #calculate class-wise balance factor from ratios of classes
    cwb = [0]*len(counts)
    for i in range(len(counts)):
        v_tot += 1/math.sqrt((counts[i]/tot))
    for index in range(len(counts)):
        v = 1/math.sqrt((counts[index]/tot))
        cwb[index] = len(counts) * (v / v_tot)
    #returning weights as a tensor
    return torch.cuda.FloatTensor(cwb)

def calc_alph(beta=0.9):
    # counts = [0] * len(learn2.dls.vocab)
    # i = 0
    # tot = 0
    # for folder in os.listdir(os.getcwd() + "/Trains/train"):
    #     counts[i] = len(os.listdir(os.getcwd() + "/Trains/train/" + folder))
    #     tot += counts[i]
    #     i += 1
    # weights = []
    # for count in counts:
    #     b = (count/tot)
    #     weights.append( (1 - b) / (len(counts)-1))
    counts = [2000,1000]
    tot = 3000
    weights = []
    for count in counts:
        b = (count / tot)
        weights.append( (1 - b) / (len(counts)-1))
    return torch.cuda.FloatTensor(weights)

def set_seed(dls,x=42): #must have dls, as it has an internal random.Random
    random.seed(x)
    dls.rng.seed(x) #added this line
    np.random.seed(x)
    torch.manual_seed(x)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(x)
        torch.cuda.manual_seed(x)

labels = os.listdir("Trains/test") #class labeled derived from folder names

path = "Trains" #85/15 split on train test data
#files = get_image_files(path) #getting all filenames for snake_images folder
batch_size = 16 #normally 32

#setting to 2nd GPU (note: if device has no 2nd GPU, error will raise)
torch.cuda.set_device(1)
torch.cuda.empty_cache()

dls = ImageDataLoaders.from_folder(path, train='train', valid='test', bs=batch_size, item_tfms=Resize(224), seed=42) #image dataloader
dls.device = default_device()
set_seed(dls)
#dls.show_batch()
weight = calc_cwb()
#plt.show()
mod_name = 'vit_base_patch16_224' #specifying model type - ViT with patch size of 16 and image size 224
learn = fastai.vision.learner.vision_learner(dls, mod_name, metrics=error_rate, loss_func=CrossEntropyLossFlat())  # 'vit_base_patch16_224'

#learn.dls.rng.seed(42)
set_seed(learn.dls)
# learn = fastai.vision.learner.vision_learner(dls, 'vit_base_patch16_224_sam', metrics=error_rate)
# print(learn.lr_find())
# plt.show()
epochs = 1
lr = 1e-3
learn.loss_func = FocalLossFlat(weight=weight)
print(learn.loss_func)
learn.fine_tune(epochs, lr)  # 0.001-0.002
learn.show_results()
print("epochs: " + str(epochs) + ", lr: " + str(lr))
interp = Interpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15, 10))
plt.show()

#presenting confusion matrix for initial training performance
class_int = ClassificationInterpretation.from_learner(learn)
class_int.plot_confusion_matrix()
plt.show()
learn.export("train_"+str(mod_name) + str(epochs) + 'lr' + str(lr) + 'bs' +str(batch_size) +'.pkl') #exporting