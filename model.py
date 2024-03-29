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

#Model Define
class BaseModel(nn.Module):
    def __init__(self, le, model_name="convnext_small_384_in22ft1k"):
        super(BaseModel, self).__init__()

        self.model = timm.create_model(model_name,pretrained=True,num_classes=len(le.classes_))
        # https://github.com/huggingface/pytorch-image-models/blob/326ade299983a1d72b0f4def1299da1bb0f6b6f2/results/results-imagenet.csv
        # resnet~
        # efficientnet~
        # vit~

        
    def forward(self, x):
        # x = self.backbone(x)
        # x = self.classifier(x)
        x = self.model(x)
        return x
