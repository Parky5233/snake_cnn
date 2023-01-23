import gc
import fastai.vision.learner
import timm
import os
import matplotlib.pyplot as plt
import numpy as np
import PIL
from fastai.vision.all import *
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

#gc.collect()
# Note: this code does not run on Windows OS
'''
This code was written following various tutorials and examples:
- https://docs.fast.ai/tutorial.vision
- https://pypi.org/project/timm/#models
'''

set_seed(42, True) #setting random seed for reproducability

'''
Aug 10
- Work on more hyperparameters (batch sizes) - in  prog
    * mod: reg, sam
    * bs: 32,64,128
    * e: 5, 15
- try to finetune the final model with the small snake data to see performance
- make duplicates of the images with slightly different crops and try to change batch size to 4 or pretrain with batch size 4
and analyze impact
'''
labels = os.listdir("snake_images/test") #class labeled derived from folder names

path = "snake_images"
#files = get_image_files(path) #getting all filenames for snake_images folder
batch_size = 16 #normally 32
dls = ImageDataLoaders.from_folder(path, train='train', valid='test', bs=batch_size, item_tfms=Resize(224), seed=42) #image dataloader

#setting to 2nd GPU (note: if device has no 2nd GPU, error will raise)
torch.cuda.set_device(1)
dls.device = default_device()

dls.rng.seed(42) #setting dataloader random seed for reproducability
#dls.show_batch()
#plt.show()
mod_name = 'vit_base_patch16_224' #specifying model type - ViT with patch size of 16 and image size 224
learn = fastai.vision.learner.vision_learner(dls, mod_name, metrics=error_rate)  # 'vit_base_patch16_224'
learn.dls.rng.seed(42)
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
class_int = ClassificationInterpretation.from_learner(learn)
class_int.plot_confusion_matrix()
plt.show()

learn.export(str(mod_name) + str(epochs) + 'lr' + str(lr) + 'bs' +str(batch_size) +'.pkl') #exporti


#fine-tuning model to small snake dataset
#learn.dls = ImageDataLoaders.from_folder("formatted_data_1", train='Train', valid='Test', bs=batch_size, item_tfms=Resize(224), seed=42) #loading in data for fine-tuning



#fine-tuning model to small snake dataset
learn2 = load_learner(os.path.join(os.getcwd()+'/snake_images',str(mod_name) + str(epochs) + 'lr' + str(lr) + 'bs' +str(batch_size) +'.pkl'))
learn2.dls = ImageDataLoaders.from_folder("formatted_data_updated", train='Train', valid='Test', bs=batch_size, item_tfms=Resize(224), seed=42) #loading in data for fine-tuning
#setting to second gpu and setting random seed
torch.cuda.set_device(1)
learn2.dls.device = default_device()
learn2.dls.rng.seed(42)
secondary_epochs = 4
#Need to figure out how to reduce to 1-2 epochs without it throwing an error
learn2.fine_tune(secondary_epochs,lr) #fine-tuning model, freeze for 1 epoch then unfreeze and run epochs
learn2.show_results()
print("Model Loaded: "+str(os.path.join(os.getcwd()+'/snake_images',str(mod_name) + str(epochs) + 'lr' + str(lr) + 'bs' +str(batch_size) +'.pkl')))
print("epochs: " + str(secondary_epochs) + ", lr: " + str(lr) +", bs: " +str(batch_size))

#presenting confusion matrix
class_int = ClassificationInterpretation.from_learner(learn2)
class_int.plot_confusion_matrix()
plt.show()

learn2.export(str(mod_name) + str(epochs) + 'lr' + str(lr) + 'bs' +str(batch_size) +'small_data.pkl') #saving fine-tuned model