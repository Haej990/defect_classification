#import
import random
import pandas as pd
import numpy as np
import os
import re
import glob 
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models
import timm
#TODO: search sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings(action='ignore') 

import wandb
wandb.init(project="dacon_papering_defect_classification")
wandb.run.log_code(".")
wandb.save("./baseline_submit.csv", policy="end")

from utils import seed_everything
from data import get_loader
from model import BaseModel

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#Hyperparameter Setting
CFG = {
    'IMG_SIZE':384, #H,W=224
    'EPOCHS':30, # if use data augmentation, need to increase
    'LEARNING_RATE':1e-2, # When using SGD, use 10-100x higher learning rate 
    'BATCH_SIZE':32, 
    'SEED':41 
}

seed_everything(CFG['SEED']) # Seed 고정


#Train
def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device) # send to GPU for training
    class_weight = [1.0] * 19 
    classes_with_small_number = [0,4,8,12,13,14,16]# 0 4 8 12 13 14 16
    for c in classes_with_small_number:
        class_weight[c] = 3.0

    # dog 100
    # cat 10
    # 1:10

    #list to tensor
    class_weight = torch.FloatTensor(class_weight)
    
    criterion = nn.CrossEntropyLoss(weight = class_weight).to(device) # # send to GPU for training 
    
    best_score = 0
    best_loss = 10000000
    
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train() # train mode
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader)): #img: input data, labels: ground truth output data #tqdm : 진행바
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            output = model(imgs) # prediction, [B, 19]
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())

            if scheduler is not None:
                scheduler.step()

        # 1 epoch of training end
        # validation begin
                    
        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val Weighted F1 Score : [{_val_score:.5f}]')
        wandb.log({'train_loss': _train_loss, 'val_loss': _val_loss, 'val_f1_score': _val_score})
            
        if best_score < _val_score:
            best_score = _val_score
            torch.save(model.state_dict(), "best_val_f1_model.pt")
            print("Saved best f1 model, epoch: {}".format(epoch))
        
        if best_loss > _val_loss:
            best_loss = _val_loss
            torch.save(model.state_dict(), "best_val_loss_model.pt")
            print("Saved best loss model, epoch: {}".format(epoch))
        
#Valid
def validation(model, criterion, val_loader, device):
    model.eval() # evaluation mode
    val_loss = []
    preds, true_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            
            pred = model(imgs)
            
            loss = criterion(pred, labels)
            
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += labels.detach().cpu().numpy().tolist()
            
            val_loss.append(loss.item())
        
        _val_loss = np.mean(val_loss)
        _val_score = f1_score(true_labels, preds, average='weighted')
        print(confusion_matrix(true_labels, preds))
        print( classification_report(true_labels, preds, target_names=le.inverse_transform([i for i in range(19)])))
    
    return _val_loss, _val_score

if __name__=="__main__":
#Run

    # get dataloader for train/valid
    train_loader, val_loader, le = get_loader(CFG)

    # model
    model = BaseModel(le)

    optimizer = torch.optim.SGD(params = model.parameters(), lr = CFG["LEARNING_RATE"]) # Adam / AdamW / SGD
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = len(train_loader)*CFG['EPOCHS'], verbose = True)
    # 무조건 웬만하면 cosine annealing씀

    train(model, optimizer, train_loader, val_loader, scheduler, device)
    #infer_model == best model == model with best validation performance
