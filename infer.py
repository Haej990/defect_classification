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
from data import CustomDataset, get_transforms, get_loader
from model import BaseModel
from train import CFG

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Inference
test = pd.read_csv('./test.csv')
_, test_transform = get_transforms(CFG)
_,_,le = get_loader(CFG)
test_dataset = CustomDataset(test['img_path'].values, None, test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=16)

def inference(model, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)

            pred = model(imgs)

            preds += pred.argmax(1).detach().cpu().numpy().tolist() #argmax == max의 위치

    preds = le.inverse_transform(preds)
    return preds

# load model
infer_model = BaseModel(le)
infer_model.load_state_dict(torch.load("best_val_loss_model.pt"))
infer_model.eval()
infer_model.to(device)

preds = inference(infer_model, test_loader, device)


submit = pd.read_csv('./sample_submission.csv')
submit['label'] = preds
submit.to_csv('./baseline_submit.csv', index=False)
