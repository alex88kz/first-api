#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import torch
import tarfile
import torchvision
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import random_split
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_url
from sklearn.model_selection import train_test_split
from pdf2image import convert_from_path
from shutil import copyfile
import shutil
import pandas as pd
import numpy as np
from collections import Counter
# get_ipython().run_line_magic('matplotlib', 'inline')


# ### Script takes thee arguments:
#    * path- the path to image we want to predict
#    * model_path - the path to trained model
#    * device - select the type of device for inference ('cpu' or 'cuda')

# In[2]:


#path = '/home/ocr/Pictures/CLASS_1_VID/vid_315_2.jpg'
path = sys.argv[1]

device = torch.device('cpu')
model_path = 'resnet_50_all.pth'


# ### Classes for dataset creation

# In[3]:


def get_train_transform():
    return transforms.Compose([
    
        transforms.Resize((224,224)), #becasue vgg takes 150*150
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.7137, 0.6974, 0.6911), (0.2814, 0.2879, 0.2976))

    ])

#Augmentation is not done for test/validation data.
def get_val_transform():
    return transforms.Compose([
    
    transforms.Resize((224,224)), #becasue vgg takes 150*150
    transforms.ToTensor(),
    transforms.Normalize((0.7137, 0.6974, 0.6911), (0.2814, 0.2879, 0.2976))
    
])


# In[4]:


class Doc_Dataset(Dataset):
    
    def __init__(self, imgs,labels, class_to_int, mode = "train", transforms = None):
        
        super().__init__()
        self.imgs = imgs
        self.labels = labels
        self.class_to_int = class_to_int
        self.mode = mode
        self.transforms = transforms
        
    def __getitem__(self, idx):
        
        image_name = self.imgs[idx]
        
        ### Reading, converting and normalizing image
        img = Image.open(image_name)
        img = img.convert('RGB')
        img = img.resize((224, 224))
        
        if self.mode == "train" or self.mode == "val":
        
            ### Preparing class label
            label = self.class_to_int[self.labels[idx]]
            label = torch.tensor(label, dtype = torch.long)
            ### Apply Transforms on image
            img = self.transforms(img)

            return img, label
        
        elif self.mode == "test":
            
            ### Apply Transforms on image
            img = self.transforms(img)

            return img
            
        
    def __len__(self):
        return len(self.imgs)


# ### Classes for model and inference

# In[5]:


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


# In[6]:


class Doc_RESNET_model(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = model_resnet50
    
    def forward(self, xb):
        return self.network(xb)


# In[7]:


# def get_default_device():
#     '''Pick GPU if available, else CPU'''
#     if torch.cuda.is_available():
#         return torch.device('cuda')
#     else:
#         return torch.device('cpu')


# In[8]:


def to_device(data,device):
    '''move tensors to chosen device'''
    if isinstance(data, (list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking = True)


# In[9]:


def make_inference(input, model,device,inv_map):
    input = to_device(input,device)
    inputs = input.unsqueeze(0)
    predictions = model(inputs)
    prediction = predictions[0].detach().cpu()
    pred_label = np.argmax(prediction)
    pred_label = int(pred_label)
    pred_label = inv_map[pred_label]
    print(pred_label)
    return pred_label


# In[10]:


# device = get_default_device()
# print(device)


# ### Predict single image from path

# In[11]:


model = torch.load(model_path, map_location="cpu")
model.eval()
model = to_device(model, device)


# In[12]:


class_to_int = {"udo" : 0, "vid" : 1,'pass' : 2,'other':3}
inv_map = {v: k for k, v in class_to_int.items()}


# In[13]:


# path = '/home/ocr/Pictures/CLASS_2_PASS/pass_31_1.jpg'
label = 'vid'
inf_image = [path]
inf_label = [label]


# In[14]:


inference_dataset = Doc_Dataset(inf_image, inf_label,class_to_int, mode = "val", transforms = get_val_transform())


# In[15]:


for i,img in enumerate(inference_dataset):
    plt.imshow(img[0].permute(1,2,0))
    answer = make_inference(img[0],model,device,inv_map)


# In[16]:


answer


# In[17]:


#get_ipython().system('pip list')

