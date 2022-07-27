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

#Note: this code does not run on Windows OS
'''
This code was written following various tutorials and examples:
- https://docs.fast.ai/tutorial.vision
- https://pypi.org/project/timm/#models

To Do:
- finetune model on train set, test model on test set and print confusion matrix
- expand analysis to sam models and others

Currently:
- pretrained vit_base_patch16_224 is being tested

CWA: (bs: 32, lr:0.0008)
- banded_watersnake: 76.44
- boa_constrictor: 90.22
- coachwhip: 86.67
- eastern_copperhead: 88.89
- eastern_milkesnake: 86.22
- grass_snake: 82.22
- gray_ratsnake: 80.0
- nothern_cottonmouth: 70.67
- nothern_watersnake: 72.00
- red_bellied_snake: 88.44
- rough_greensnake: 99.11
- western_ribbon_snake: 92.00

Overall: 84.4% accuracy

Note: was achieving better with bs:64 (default bs) but randomly started to get out of cuda memory errors
'''

def set_seed(x=42):
    random.seed(x)
    np.random.seed(x)
    torch.manual_seed(x)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(x)

set_seed()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#print(timm.list_models('*vit*'))

model = timm.create_model('vit_base_patch16_224', pretrained=True,num_classes=12)

#print(device)

labels = os.listdir("snake_images/test")

path = "snake_images"
files = get_image_files(path)

pat = r'^(.*)_\d+.jpg'
print(os.listdir())
dls = ImageDataLoaders.from_folder(path, train='train',valid='test',bs=32, item_tfms=Resize(224))
dls.show_batch()
plt.show()

learn = fastai.vision.learner.vision_learner(dls, 'vit_base_patch16_224_sam', metrics=error_rate)#'vit_base_patch16_224'

#learn = fastai.vision.learner.vision_learner(dls, 'vit_base_patch16_224_sam', metrics=error_rate)
print(learn.lr_find())#0.0005
plt.show()

learn.fine_tune(2,8e-4)
learn.show_results()

interp = Interpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,10))
plt.show()

class_int = ClassificationInterpretation.from_learner(learn)

class_int.plot_confusion_matrix()
plt.show()

learn.export('pretrained_sam_vit.pkl')