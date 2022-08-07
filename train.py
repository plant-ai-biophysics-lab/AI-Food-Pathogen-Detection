#%%
#=========================================== Define local file path ===========================================#

## For model training: A path to input images & a pretrained model
root_path = '/data2/ai-food-pathogen-data/'
pretrained_model_path = '/data/jyyi/ai-food-pathogen-results/best_models/7 best_model.pth'

## For model fine-tuning: A path to our model trained on multiple bacterial species
model_3_path = '/data/jyyi/ai-food-pathogen-results/best_models/best_model220505_tr3.pth'

## A path to save training log files
runs_path = '/data/jyyi/ai-food-pathogen-results/runs'

#==============================================================================================================# 

import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import MicrobeDataset, get_transform
import models
import utils
import engine
from engine import train_one_epoch, validate#, evaluate
  
'''
Load train/validation datasets & model

'''
dataset_name = ['train/mono-sp/mono_1', 'train/mono-sp/mono_2', 'train/mono-sp/mono_3',  #0-2
                'val/mono-sp/mono_1', 'val/mono-sp/mono_2', 'val/mono-sp/mono_3',        #3-5
                'train/mono_1', 'train/mono_2-3', 'val/mono_1', 'val/mono_2-3',          #6-9
                'train/non-ecoli', 'val/non-ecoli',                                      #10-11
                'train/labmix_0', 'train/labmix_1', 'train/labmix_2-3',                  #12-14
                'val/labmix_0', 'val/labmix_1', 'val/labmix_2-3'                         #15-17                
                ]

## Initialize empty lists
imgpath = [[] for _ in range(len(dataset_name))]
annotpath = [[] for _ in range(len(dataset_name))]
dataset = [[] for _ in range(len(dataset_name))]

## Load datasets via MicrobeDataset
for i in range(len(dataset_name)): # for each directory
    
    ## Image and labels
    imgpath[i] = root_path+dataset_name[i]+'/'
    if 'non-ecoli' in dataset_name[i]:
        annotpath[i] = None
    elif 'mono_2' in dataset_name[i] or 'mono_3' in dataset_name[i]:
        annotpath[i] = root_path+'mono_2-3_annotations.xml'
    else:
        annotpath[i] = root_path+dataset_name[i].split('/')[-1]+'_annotations.xml'
    
    ## Transforms
    if 'train' in dataset_name[i]:
        transforms = get_transform(True)
    else:
        transforms = get_transform(False)     

    dataset[i] = MicrobeDataset(root=imgpath[i], annot_path=annotpath[i], transforms=transforms, n_anchors=100)      

## E. coli monoculture (split dataset)
trainset_2s = torch.utils.data.ConcatDataset([dataset[0],dataset[1],dataset[2]])
valset_2s = torch.utils.data.ConcatDataset([dataset[3],dataset[4],dataset[5]])

## E. coli monoculture
trainset_2 = torch.utils.data.ConcatDataset([dataset[6],dataset[7]])
valset_2 = torch.utils.data.ConcatDataset([dataset[8],dataset[9]])

## E. coli + non-E. coli
trainset_3 = torch.utils.data.ConcatDataset([dataset[6],dataset[7],dataset[10]])
valset_3 = torch.utils.data.ConcatDataset([dataset[8],dataset[9],dataset[11]])

## Microbial mixture
trainset_33 = torch.utils.data.ConcatDataset([dataset[12],dataset[13],dataset[14]])
valset_33 = torch.utils.data.ConcatDataset([dataset[15],dataset[16],dataset[17]])

## Load Faster R-CNN model
model_loader = models.ModelLoader()
model = model_loader.get_model(model_name='faster_rcnn', num_classes=2,
                               pretrained_backbone=True, # returns a model with backbone pre-trained on ImageNet
                               pretrained_model_path=pretrained_model_path
                               )

#%%
'''
Select train/val datasets to be used for model training

'''
## Uncomment one of the following three lines
trainset, valset = trainset_2s, valset_2s
# trainset, valset = trainset_2, valset_2
# trainset, valset = trainset_3, valset_3

#%%
'''
For fine-tuning, uncomment this cell

'''
# trainset, valset = trainset_33, valset_33
# model.load_state_dict(torch.load(model_3_path))

#%%
'''
Model training

'''
## Initialize dataloaders
dataloaders = {
    'train': DataLoader(trainset, batch_size=4, num_workers=0, shuffle=True, collate_fn=utils.collate_fn),
    'val': DataLoader(valset, batch_size=4, shuffle=False, num_workers=0, collate_fn=utils.collate_fn),
    }

## Model setup
device=torch.device('cuda')
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)

best_val_loss = 999
num_epochs = 300
val_size = len(valset)

for epoch in range(num_epochs):
    ## Train for one epoch, printing every 10 iterations
    writer = SummaryWriter(log_dir=runs_path)
    train_one_epoch(model, optimizer, dataloaders['train'], device, epoch, print_freq=2)
   
    ## Evaluate and checkpoint using the validate dataset
    total_val_loss, best_val_loss = validate(model, dataloaders['val'], device=device, best_val_loss=best_val_loss)
    writer.add_scalar("val loss", total_val_loss/val_size, epoch)
    writer.flush()
    

# %%
