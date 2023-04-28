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


#CustomDataset

class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        
    def __getitem__(self, index):
        img_path = self.img_path_list[index] # image path == 파일 경로
        
        image = cv2.imread(img_path) # cv2: opencv 라는 library 
                                     # imread == imageread == 파일경로에서 이미지를 읽어서, 이걸 H x W x 3 으로 읽어옴
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image'] # 데이터 변환 (0-255 -> 0-1로 바꿈) 및 증강 (있으면) 진행
        
        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.img_path_list)

def get_transforms(CFG):

    train_transform = A.Compose([
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']), # 다 244 244로 통일== 왜? 그래야 batchify 가 가능
                            #A.Rotate(),
                            A.ColorJitter(p=1.0),
                            A.RandomRotate90(p=1.0),
                            A.Affine(scale=(0.8,1.2), translate_percent=(0.0,0.3), shear=[-45,45]),
                            A.Cutout(num_holes=12, max_h_size=10, max_w_size=10, fill_value=0, always_apply=False, p=0.5),
                            #A.AdvancedBlur(),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0), #0-255를 0-1로 바꾸는 과정 : 데이터셋에 맞춰서 변환해줌
                            # TODO: data augmentation can come here
                            ToTensorV2() #tensor == 딥러닝용 단어 matrix
                            # TODO: or here
                            ])
                            #https://albumentations.ai/docs/api_reference/augmentations/

    test_transform = A.Compose([ # valid/test의 경우 data augmentation안해줌
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])
                            
    return train_transform, test_transform

def get_loader(CFG):

    #Data Pre-processing

    all_img_list = glob.glob('./train/*/*') #glob == 파일 경로 불러오기 

    df = pd.DataFrame(columns=['img_path', 'label'])
    df['img_path'] = all_img_list
    df['label'] = df['img_path'].apply(lambda x : str(x).split('/')[2]) #TODO: lambda 찾아보기



    train, val, _, _ = train_test_split(df, df['label'], test_size=0.05, stratify=df['label'], random_state=CFG['SEED']) # split train, valid 
                                                                                                                        # valid 고정 -> seed 고정 == randomness 고정

    # 반점 하나 옮기기
    if len(val.loc[val.label=='반점']) == 0:
        cond0 = train.label == '반점'
        rows = train.loc[cond0]
        to_move = rows.iloc[0]
        val = val.append(to_move, ignore_index=True)
        train = train.drop(index=rows.index[0])

    # 틈새과다 하나 옮기기

    if len(val.loc[val.label=='틈새과다']) == 0:
        cond1 = train.label == '틈새과다'
        rows = train.loc[cond1]
        to_move = rows.iloc[0]
        val = val.append(to_move, ignore_index=True)
        train = train.drop(index=rows.index[0])

    # data balancing
    multiplier = {'훼손':1, '오염':2, '걸레받이수정':4, '꼬임':6, '터짐':7, '곰팡이':7, '오타공':7, '몰딩수정':7,'면불량':12, '석고수정':20, '들뜸':20, '피스':20, '창틀,문틀수정':45, '울음':45, '이음부불량':70, '녹오염':85, '가구수정':85, '반점':600, '틈새과다':300 } 
    reps = [multiplier[v] for v in train.label] # TODO: list comprehension 
    train = train.loc[np.repeat(train.index.values, reps)]
    #Label-Encoding
    le = preprocessing.LabelEncoder() #label -> 숫자로 바꿔주는 과정
    train['label'] = le.fit_transform(train['label'])
    val['label'] = le.transform(val['label'])

    train_transform, test_transform = get_transforms(CFG)

    train_dataset = CustomDataset(train['img_path'].values, train['label'].values, train_transform)
    train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=16) #shuffle == 데이터 순서를 섞는다. 학습할 땐 True가 좋음

    val_dataset = CustomDataset(val['img_path'].values, val['label'].values, test_transform)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=16) 

    return train_loader, val_loader, le