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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#Hyperparameter Setting

CFG = {
    'IMG_SIZE':224, #H,W=224
    'EPOCHS':20, # if use data augmentation, need to increase
    'LEARNING_RATE':3e-2, # When using SGD, use 10-100x higher learning rate 
    'BATCH_SIZE':64, 
    'SEED':41 
}

#Fixed RandomSeed
# reproducibility 
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정

#Data Pre-processing

all_img_list = glob.glob('./train/*/*') #glob == 파일 경로 불러오기 

df = pd.DataFrame(columns=['img_path', 'label'])
df['img_path'] = all_img_list
df['label'] = df['img_path'].apply(lambda x : str(x).split('/')[2]) #TODO: lambda 찾아보기



train, val, _, _ = train_test_split(df, df['label'], test_size=0.1, stratify=df['label'], random_state=CFG['SEED']) # split train, valid 
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

train_transform = A.Compose([
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']), # 다 244 244로 통일== 왜? 그래야 batchify 가 가능
                            #A.Rotate(),
                            A.ColorJitter(p=1.0),
                            A.RandomRotate90(p=1.0),
                            A.Affine (scale=(0.8,1.2), translate_percent=(0.0,0.05), shear=[-45,45]),
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

train_dataset = CustomDataset(train['img_path'].values, train['label'].values, train_transform)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=16) #shuffle == 데이터 순서를 섞는다. 학습할 땐 True가 좋음

val_dataset = CustomDataset(val['img_path'].values, val['label'].values, test_transform)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=16) 

#Model Define
class BaseModel(nn.Module):
    def __init__(self, num_classes=len(le.classes_)):
        super(BaseModel, self).__init__()
        # self.backbone = models.efficientnet_b0(pretrained=True) # ImageNet pretrained
                                                                 # #classes ==1000
                                                                
        # self.classifier = nn.Linear(1000, num_classes) # 1000 -> 19

        self.model = timm.create_model('tf_efficientnet_b3_ns',pretrained=True,num_classes=num_classes)
        # https://github.com/huggingface/pytorch-image-models/blob/326ade299983a1d72b0f4def1299da1bb0f6b6f2/results/results-imagenet.csv
        # resnet~
        # efficientnet~
        # vit~

        
    def forward(self, x):
        # x = self.backbone(x)
        # x = self.classifier(x)
        x = self.model(x)
        return x

#Train

def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device) # send to GPU for training 
    criterion = nn.CrossEntropyLoss().to(device) # # send to GPU for training 
    
    best_score = 0
    best_model = None
    
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train() # train mode
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader)): #img: input data, labels: ground truth output data
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            output = model(imgs) # prediction, [B, 19]
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())

        # 1 epoch of training end
        # validation begin
                    
        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val Weighted F1 Score : [{_val_score:.5f}]')
        wandb.log({'train_loss': _train_loss, 'val_loss': _val_loss, 'val_f1_score': _val_score})
        if scheduler is not None:
            scheduler.step(_val_score)
            
        if best_score < _val_score:
            best_score = _val_score
            best_model = model
    
    return best_model

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


#Run

model = BaseModel()

optimizer = torch.optim.SGD(params = model.parameters(), lr = CFG["LEARNING_RATE"]) # Adam / AdamW / SGD
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)
# 무조건 웬만하면 cosine annealing씀

infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)
#infer_model == best model == model with best validation performance
###################################################################################
# Inference
test = pd.read_csv('./test.csv')

test_dataset = CustomDataset(test['img_path'].values, None, test_transform)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

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

preds = inference(infer_model, test_loader, device)


submit = pd.read_csv('./sample_submission.csv')
submit['label'] = preds
submit.to_csv('./baseline_submit.csv', index=False)
