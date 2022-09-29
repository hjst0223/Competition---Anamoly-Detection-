# Library


```python
import os
import cv2
import time
import random
import logging  # 로그 출력
import easydict  # 속성으로 dict 값에 access할 수 있음
import numpy as np
import pandas as pd
from tqdm import tqdm  # process bar
from os.path import join as opj
from ptflops import get_model_complexity_info
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import timm
import torch
import torch.nn as nn
import torch_optimizer as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, grad_scaler
from torchvision import transforms

import warnings
warnings.filterwarnings('ignore')
```

# Config

Hyper-parameter 정의


```python
args = easydict.EasyDict(
    {'exp_num':'0',
     
     # Path settings
     'data_path':'./data',
     'Kfold':5,
     'model_path':'results/',

     # Model parameter settings 
     'encoder_name':'regnety_160',
     'drop_path_rate':0.2,
     
     # Training parameter settings
     ## Base Parameter
     'img_size':224,
     'batch_size':16,
     'epochs':200,
     'optimizer':'Lamb',
     'initial_lr':5e-6,
     'weight_decay':1e-3,

     ## Augmentation
     'aug_ver':2,

     ## Scheduler (OnecycleLR)
     'scheduler':'cycle',
     'warm_epoch':5,
     'max_lr':1e-3,

     ### Cosine Annealing
     'min_lr':5e-6,
     'tmax':145,

     ## etc.
     'patience':50,
     'clipping':None,

     # Hardware settings
     'amp':True,
     'multi_gpu':False,
     'logging':False,
     'num_workers':0,
     'seed':42
    })
```

# Utils for training and Logging


```python
# Warmup Learning rate scheduler
from torch.optim.lr_scheduler import _LRScheduler
class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimizer(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

# Logging
def get_root_logger(logger_name='basicsr',
                    log_level=logging.INFO,
                    log_file=None):

    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger.hasHandlers():
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(format=format_str, level=log_level)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger

class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)
```

# Data Preprocessing
- 원본 이미지 사이즈보다 작은 (256,256)로 resize하여 데이터를 새롭게 저장


```python
df = pd.read_csv('./data/train_df.csv')

# Resize Train Images
save_path = './data/train_256_new'  # 새로 저장할 폴더 경로
os.makedirs(save_path, exist_ok=True)
for img in tqdm(df['file_name']):  # train_df의 'file_name' 컬럼을 참고하여
    name = os.path.basename(img)
    img = cv2.imread(opj('./data/train/', img))  # 해당 경로에 있는 png 이미지 읽어서
    img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
    img = cv2.imwrite(opj(save_path, name), img)  # 새 폴더에 저장

# Resize Test Images
df = pd.read_csv('./data/test_df.csv')
save_path = './data/test_256_new'
os.makedirs(save_path, exist_ok=True)
for img in tqdm(df['file_name']):
    name = os.path.basename(img)
    img = cv2.imread(opj('./data/test/', img))
    img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
    img = cv2.imwrite(opj(save_path, name), img)
```

# Dataset & Loader


```python
class Train_Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.file_name = df['file_name'].values
        # 각 label을 str->index로 변환
        labels = ['bottle-broken_large', 'bottle-broken_small', 'bottle-contamination', 'bottle-good', 'cable-bent_wire', 'cable-cable_swap', 'cable-combined', 'cable-cut_inner_insulation', 'cable-cut_outer_insulation', 'cable-good', 'cable-missing_cable', 'cable-missing_wire', 'cable-poke_insulation', 'capsule-crack', 'capsule-faulty_imprint', 'capsule-good', 'capsule-poke', 'capsule-scratch', 'capsule-squeeze', 'carpet-color', 'carpet-cut', 'carpet-good', 'carpet-hole', 'carpet-metal_contamination', 'carpet-thread', 'grid-bent', 'grid-broken', 'grid-glue', 'grid-good', 'grid-metal_contamination', 'grid-thread', 'hazelnut-crack', 'hazelnut-cut', 'hazelnut-good', 'hazelnut-hole', 'hazelnut-print', 'leather-color', 'leather-cut', 'leather-fold', 'leather-glue', 'leather-good', 'leather-poke', 'metal_nut-bent', 'metal_nut-color', 'metal_nut-flip', 'metal_nut-good', 'metal_nut-scratch', 'pill-color', 'pill-combined', 'pill-contamination', 'pill-crack', 'pill-faulty_imprint', 'pill-good', 'pill-pill_type', 'pill-scratch', 'screw-good', 'screw-manipulated_front', 'screw-scratch_head', 'screw-scratch_neck', 'screw-thread_side', 'screw-thread_top', 'tile-crack', 'tile-glue_strip', 'tile-good', 'tile-gray_stroke', 'tile-oil', 'tile-rough', 'toothbrush-defective', 'toothbrush-good', 'transistor-bent_lead', 'transistor-cut_lead', 'transistor-damaged_case', 'transistor-good', 'transistor-misplaced', 'wood-color', 'wood-combined', 'wood-good', 'wood-hole', 'wood-liquid', 'wood-scratch', 'zipper-broken_teeth', 'zipper-combined', 'zipper-fabric_border', 'zipper-fabric_interior', 'zipper-good', 'zipper-rough', 'zipper-split_teeth', 'zipper-squeezed_teeth']
        new = dict(zip(range(len(labels)),labels))
        label_decoder = {val:key for key, val in new.items()}
        df['label'] = df['label'].replace(label_decoder)

        self.target = df['label'].values  # 목표는 label
        self.transform = transform

        print(f'Dataset size:{len(self.file_name)}')

    def __getitem__(self, idx):  # train 경로에 있는 png 이미지 읽어서 float32로 변환
        image = cv2.imread(opj('./data/train_256_new/', self.file_name[idx])).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # BGR=>RGB 변환

        target = self.target[idx]

        if self.transform is not None:
        # HWC => CHW-layout 변환
            image = self.transform(torch.from_numpy(image.transpose(2,0,1)))

        return image, target

    def __len__(self):
        return len(self.file_name)

class Test_dataset(Dataset):
    def __init__(self, df, transform=None):
        self.test_file_name = df['file_name'].values
        self.transform = transform

        print(f'Test Dataset size:{len(self.test_file_name)}')

    def __getitem__(self, idx): # test 경로에 있는 png 이미지 읽어서 float32로 변환
        image = cv2.imread(opj('./data/test_256_new/', self.test_file_name[idx])).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # BGR=>RGB 변환

        if self.transform is not None:
            image = self.transform(torch.from_numpy(image.transpose(2,0,1)))

        return image

    def __len__(self):
        return len(self.test_file_name)

def get_loader(df, phase: str, batch_size, shuffle,
               num_workers, transform):
    if phase == 'test':
        dataset = Test_dataset(df, transform)  
        # num_workers : 데이터 로딩에 사용하는 subprocess 개수
        # pin_memory : True - 데이터로더가 Tensor를 CUDA 고정 메모리에 올림
        # drop_last : batch의 크기에 따른 의존도 높은 함수를 사용할 때 우려되는 경우 마지막 batch를 사용하지 않을 수 있음
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    else:
        dataset = Train_Dataset(df, transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                                 drop_last=False)
    return data_loader

def get_train_augmentation(img_size, ver):
    if ver == 1: # for validset
        transform = transforms.Compose([
#                 transforms.Resize((img_size, img_size)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                ])

    if ver == 2:
        transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomCrop(224),
                transforms.RandomPerspective(),
                transforms.RandomAffine((20)),  # x, y축으로 이미지 늘림
                transforms.RandomRotation(90),
#                 transforms.Resize((img_size, img_size)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
    
    return transform
```

# Network


```python
class Network(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 사전 학습된 모델 사용하기
        self.encoder = timm.create_model(args.encoder_name, pretrained=True,
                                    drop_path_rate=args.drop_path_rate,
                                    )
        
        if 'regnet' in args.encoder_name:        
            num_head = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Linear(num_head, 88)
        
        elif 'efficient' in args.encoder_name:
            num_head = self.encoder.classifier.in_features
            self.encoder.classifier = nn.Linear(num_head, 88)

    def forward(self, x):
        x = self.encoder(x)
        return x

class Network_test(nn.Module):
    def __init__(self, encoder_name):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=True,
                                    drop_path_rate=0,
                                    )
        
        if 'regnet' in encoder_name:        
            num_head = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Linear(num_head, 88)
        
        elif 'efficient' in encoder_name:
            num_head = self.encoder.classifier.in_features
            self.encoder.classifier = nn.Linear(num_head, 88)
    
    def forward(self, x):
        x = self.encoder(x)
        return x
```

# Trainer for Training & Validation


```python
class Trainer():
    def __init__(self, args, save_path):
        '''
        args: arguments
        save_path: Model 가중치 저장 경로
        '''
        super(Trainer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Logging
        log_file = os.path.join(save_path, 'log_0511_1.log')
        self.logger = get_root_logger(logger_name='IR', log_level=logging.INFO, log_file=log_file)
        self.logger.info(args)
        # self.logger.info(args.tag)

        # Train, Valid Set load
        ############################################################################
        df_train = pd.read_csv(opj(args.data_path, 'train_df.csv'))
        print('Read train_df.csv')

        kf = StratifiedKFold(n_splits=args.Kfold, shuffle=True, random_state=args.seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(df_train)), y=df_train['label'])):
            df_train.loc[val_idx, 'fold'] = fold
        val_idx = list(df_train[df_train['fold'] == int(args.fold)].index)

        df_val = df_train[df_train['fold'] == args.fold].reset_index(drop=True)
        df_train = df_train[df_train['fold'] != args.fold].reset_index(drop=True)

        # Augmentation
        self.train_transform = get_train_augmentation(img_size=args.img_size, ver=args.aug_ver)
        self.test_transform = get_train_augmentation(img_size=args.img_size, ver=1)

        # TrainLoader
        self.train_loader = get_loader(df_train, phase='train', batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, transform=self.train_transform)
        self.val_loader = get_loader(df_val, phase='train', batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers, transform=self.test_transform)

        # Network
        self.model = Network(args).to(self.device)
        macs, params = get_model_complexity_info(self.model, (3, args.img_size, args.img_size), as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        self.logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        self.logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))

        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer & Scheduler
        self.optimizer = optim.Lamb(self.model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
        
        iter_per_epoch = len(self.train_loader)
        self.warmup_scheduler = WarmUpLR(self.optimizer, iter_per_epoch * args.warm_epoch)

        if args.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.milestone, gamma=args.lr_factor, verbose=True)
        elif args.scheduler == 'cos':
            tmax = args.tmax # half-cycle 
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = tmax, eta_min=args.min_lr, verbose=True)
        elif args.scheduler == 'cycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.max_lr, steps_per_epoch=iter_per_epoch, epochs=args.epochs)

        if args.multi_gpu:
            self.model = nn.DataParallel(self.model).to(self.device)

        # Train / Validate
        best_loss = np.inf
        best_acc = 0
        best_epoch = 0
        early_stopping = 0
        start = time.time()
        for epoch in range(1, args.epochs+1):
            self.epoch = epoch

            if args.scheduler == 'cos':
                if epoch > args.warm_epoch:
                    self.scheduler.step()

            # Training
            train_loss, train_acc, train_f1 = self.training(args)

            # Model weight in Multi_GPU or Single GPU
            state_dict= self.model.module.state_dict() if args.multi_gpu else self.model.state_dict()

            # Validation
            val_loss, val_acc, val_f1 = self.validate(args, phase='val')

            # Save models
            if val_loss < best_loss:
                early_stopping = 0
                best_epoch = epoch
                best_loss = val_loss
                best_acc = val_acc
                best_f1 = val_f1

                torch.save({'epoch':epoch,
                            'state_dict':state_dict,
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict(),
                    }, os.path.join(save_path, 'best_model.pth'))
                self.logger.info(f'-----------------SAVE:{best_epoch}epoch----------------')
            else:
                early_stopping += 1

            # Early Stopping
            if early_stopping == args.patience:
                break
                
            print(f'\nbest epoch:{best_epoch}/loss:{best_loss:.4f}/f1:{best_f1:.4f}')

        self.logger.info(f'\nBest Val Epoch:{best_epoch} | Val Loss:{best_loss:.4f} | Val Acc:{best_acc:.4f} | Val F1:{best_f1:.4f}')
        end = time.time()
        self.logger.info(f'Total Process time:{(end - start) / 60:.3f}Minute')

    # Training
    def training(self, args):
        self.model.train()
        train_loss = AvgMeter()
        train_acc = 0
        preds_list = []
        targets_list = []

        scaler = grad_scaler.GradScaler()
        for i, (images, targets) in enumerate(tqdm(self.train_loader)):
            images = torch.tensor(images, device=self.device, dtype=torch.float32)
            targets = torch.tensor(targets, device=self.device, dtype=torch.long)
            
            if self.epoch <= args.warm_epoch:
                self.warmup_scheduler.step()

            self.model.zero_grad(set_to_none=True)
            if args.amp:
                with autocast():
                    preds = self.model(images)
                    loss = self.criterion(preds, targets)
                scaler.scale(loss).backward()

                # Gradient Clipping
                if args.clipping is not None:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clipping)

                scaler.step(self.optimizer)
                scaler.update()

            else:
                preds = self.model(images)
                loss = self.criterion(preds, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), args.clipping)
                self.optimizer.step()

            if args.scheduler == 'cycle':
                if self.epoch > args.warm_epoch:
                    self.scheduler.step()

            # Metric
            train_acc += (preds.argmax(dim=1) == targets).sum().item()
            preds_list.extend(preds.argmax(dim=1).cpu().detach().numpy())
            targets_list.extend(targets.cpu().detach().numpy())
            # log
            train_loss.update(loss.item(), n=images.size(0))

        train_acc /= len(self.train_loader.dataset)
        train_f1 = f1_score(np.array(targets_list), np.array(preds_list), average='macro')

        self.logger.info(f'Epoch:[{self.epoch:03d}/{args.epochs:03d}]')
        self.logger.info(f'Train Loss:{train_loss.avg:.3f} | Acc:{train_acc:.4f} | F1:{train_f1:.4f}')
        return train_loss.avg, train_acc, train_f1
            
    # Validation or Dev
    def validate(self, args, phase='val'):
        self.model.eval()
        with torch.no_grad():
            val_loss = AvgMeter()
            val_acc = 0
            preds_list = []
            targets_list = []

            for i, (images, targets) in enumerate(self.val_loader):
                images = torch.tensor(images, device=self.device, dtype=torch.float32)
                targets = torch.tensor(targets, device=self.device, dtype=torch.long)

                preds = self.model(images)
                loss = self.criterion(preds, targets)

                # Metric
                val_acc += (preds.argmax(dim=1) == targets).sum().item()
                preds_list.extend(preds.argmax(dim=1).cpu().detach().numpy())
                targets_list.extend(targets.cpu().detach().numpy())

                # log
                val_loss.update(loss.item(), n=images.size(0))
            val_acc /= len(self.val_loader.dataset)
            val_f1 = f1_score(np.array(targets_list), np.array(preds_list), average='macro')

            self.logger.info(f'{phase} Loss:{val_loss.avg:.3f} | Acc:{val_acc:.4f} | F1:{val_f1:.4f}')
        return val_loss.avg, val_acc, val_f1
```

# Main


```python
def main(args):
    print('<---- Training Params ---->')
    
    # Random Seed
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    save_path = os.path.join(args.model_path, (args.exp_num).zfill(3))
    
    # Create model directory
    os.makedirs(save_path, exist_ok=True)
    Trainer(args, save_path)

    return save_path
```

# Predict & Ensemble

```python
def predict(encoder_name, test_loader, device, model_path):
    model = Network_test(encoder_name).to(device)
    model.load_state_dict(torch.load(opj(model_path, 'best_model.pth'))['state_dict'])
    model.eval()
    preds_list = []
    with torch.no_grad():
        for images in tqdm(test_loader):
            images = torch.as_tensor(images, device=device, dtype=torch.float32)
            preds = model(images)
            preds = torch.softmax(preds, dim=1)
            preds_list.extend(preds.cpu().tolist())

    return np.array(preds_list)

def ensemble_5fold(model_path_list, test_loader, device):
    predict_list = []
    for model_path in model_path_list:
        prediction = predict(encoder_name= 'regnety_160', 
                             test_loader = test_loader, device = device, model_path = model_path)
        predict_list.append(prediction)
    ensemble = (predict_list[0] + predict_list[1] + predict_list[2] + predict_list[3] + predict_list[4])/len(predict_list)

    return ensemble
```

# Train & Inference


```python
img_size = 224
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sub = pd.read_csv('./data/sample_submission.csv')
df_train = pd.read_csv('./data/train_df.csv')
df_test = pd.read_csv('./data/test_df.csv')
```


```python
test_transform = get_train_augmentation(img_size=img_size, ver=1)
test_dataset = Test_dataset(df_test, test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
```

    Test Dataset size:2154
    


```python
start = 0 # first time : Only Trainset
models_path = []
for s_fold in range(5): # 5fold
    args.fold = s_fold
    args.exp_num = str(s_fold)
    save_path = main(args)
    models_path.append(save_path)
```

    2022-05-11 02:42:52,226 INFO: {'exp_num': '0', 'data_path': './data', 'Kfold': 5, 'model_path': 'results/', 'encoder_name': 'regnety_160', 'drop_path_rate': 0.2, 'img_size': 224, 'batch_size': 16, 'epochs': 200, 'optimizer': 'Lamb', 'initial_lr': 5e-06, 'weight_decay': 0.001, 'aug_ver': 2, 'scheduler': 'cycle', 'warm_epoch': 5, 'max_lr': 0.001, 'min_lr': 5e-06, 'tmax': 145, 'patience': 50, 'clipping': None, 'amp': True, 'multi_gpu': False, 'logging': False, 'num_workers': 0, 'seed': 42, 'fold': 0}
    

    <---- Training Params ---->
    Read train_df.csv
    Dataset size:3421
    Dataset size:856
    

    2022-05-11 02:42:53,006 INFO: Loading pretrained weights from url (https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth)
    2022-05-11 02:42:57,403 INFO: Computational complexity:       15.93 GMac
    2022-05-11 02:42:57,403 INFO: Number of parameters:           80.83 M 
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [03:19<00:00,  1.07it/s]
    2022-05-11 02:46:16,794 INFO: Epoch:[001/200]
    2022-05-11 02:46:16,794 INFO: Train Loss:4.484 | Acc:0.0126 | F1:0.0048
    2022-05-11 02:46:33,456 INFO: val Loss:4.537 | Acc:0.0000 | F1:0.0000
    2022-05-11 02:46:35,076 INFO: -----------------SAVE:1epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:51<00:00,  1.92it/s]
    2022-05-11 02:48:26,493 INFO: Epoch:[002/200]
    2022-05-11 02:48:26,494 INFO: Train Loss:4.470 | Acc:0.0149 | F1:0.0053
    2022-05-11 02:48:36,234 INFO: val Loss:4.526 | Acc:0.0000 | F1:0.0000
    2022-05-11 02:48:37,961 INFO: -----------------SAVE:2epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 02:50:28,236 INFO: Epoch:[003/200]
    2022-05-11 02:50:28,237 INFO: Train Loss:4.447 | Acc:0.0202 | F1:0.0049
    2022-05-11 02:50:37,931 INFO: val Loss:4.487 | Acc:0.0012 | F1:0.0002
    2022-05-11 02:50:39,735 INFO: -----------------SAVE:3epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:52<00:00,  1.90it/s]
    2022-05-11 02:52:32,487 INFO: Epoch:[004/200]
    2022-05-11 02:52:32,487 INFO: Train Loss:4.402 | Acc:0.0400 | F1:0.0108
    2022-05-11 02:52:42,557 INFO: val Loss:4.458 | Acc:0.0070 | F1:0.0013
    2022-05-11 02:52:44,349 INFO: -----------------SAVE:4epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.86it/s]
    2022-05-11 02:54:39,719 INFO: Epoch:[005/200]
    2022-05-11 02:54:39,720 INFO: Train Loss:4.351 | Acc:0.0672 | F1:0.0136
    2022-05-11 02:54:49,469 INFO: val Loss:4.395 | Acc:0.0292 | F1:0.0043
    2022-05-11 02:54:51,243 INFO: -----------------SAVE:5epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:51<00:00,  1.92it/s]
    2022-05-11 02:56:42,675 INFO: Epoch:[006/200]
    2022-05-11 02:56:42,676 INFO: Train Loss:4.023 | Acc:0.3715 | F1:0.0747
    2022-05-11 02:56:52,461 INFO: val Loss:3.713 | Acc:0.5794 | F1:0.1092
    2022-05-11 02:56:54,271 INFO: -----------------SAVE:6epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:51<00:00,  1.91it/s]
    2022-05-11 02:58:46,060 INFO: Epoch:[007/200]
    2022-05-11 02:58:46,060 INFO: Train Loss:3.233 | Acc:0.7469 | F1:0.1339
    2022-05-11 02:58:55,871 INFO: val Loss:2.734 | Acc:0.7652 | F1:0.1344
    2022-05-11 02:58:57,639 INFO: -----------------SAVE:7epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.80it/s]
    2022-05-11 03:00:56,738 INFO: Epoch:[008/200]
    2022-05-11 03:00:56,739 INFO: Train Loss:2.354 | Acc:0.8255 | F1:0.1470
    2022-05-11 03:01:06,794 INFO: val Loss:1.743 | Acc:0.8341 | F1:0.1448
    2022-05-11 03:01:08,588 INFO: -----------------SAVE:8epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:52<00:00,  1.90it/s]
    2022-05-11 03:03:01,367 INFO: Epoch:[009/200]
    2022-05-11 03:03:01,367 INFO: Train Loss:1.605 | Acc:0.8322 | F1:0.1481
    2022-05-11 03:03:11,147 INFO: val Loss:1.156 | Acc:0.8470 | F1:0.1557
    2022-05-11 03:03:12,895 INFO: -----------------SAVE:9epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:01<00:00,  1.77it/s]
    2022-05-11 03:05:13,946 INFO: Epoch:[010/200]
    2022-05-11 03:05:13,946 INFO: Train Loss:1.269 | Acc:0.8410 | F1:0.1535
    2022-05-11 03:05:25,019 INFO: val Loss:1.034 | Acc:0.8481 | F1:0.1560
    2022-05-11 03:05:27,160 INFO: -----------------SAVE:10epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:52<00:00,  1.90it/s]
    2022-05-11 03:07:19,871 INFO: Epoch:[011/200]
    2022-05-11 03:07:19,872 INFO: Train Loss:1.083 | Acc:0.8460 | F1:0.1553
    2022-05-11 03:07:30,380 INFO: val Loss:0.910 | Acc:0.8481 | F1:0.1560
    2022-05-11 03:07:32,611 INFO: -----------------SAVE:11epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.93it/s]
    2022-05-11 03:09:23,592 INFO: Epoch:[012/200]
    2022-05-11 03:09:23,592 INFO: Train Loss:0.985 | Acc:0.8460 | F1:0.1557
    2022-05-11 03:09:33,357 INFO: val Loss:0.809 | Acc:0.8481 | F1:0.1560
    2022-05-11 03:09:35,248 INFO: -----------------SAVE:12epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.93it/s]
    2022-05-11 03:11:25,853 INFO: Epoch:[013/200]
    2022-05-11 03:11:25,853 INFO: Train Loss:0.891 | Acc:0.8477 | F1:0.1559
    2022-05-11 03:11:35,609 INFO: val Loss:0.763 | Acc:0.8481 | F1:0.1560
    2022-05-11 03:11:37,314 INFO: -----------------SAVE:13epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 03:13:27,664 INFO: Epoch:[014/200]
    2022-05-11 03:13:27,664 INFO: Train Loss:0.809 | Acc:0.8480 | F1:0.1580
    2022-05-11 03:13:37,432 INFO: val Loss:0.698 | Acc:0.8505 | F1:0.1677
    2022-05-11 03:13:39,266 INFO: -----------------SAVE:14epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.83it/s]
    2022-05-11 03:15:36,280 INFO: Epoch:[015/200]
    2022-05-11 03:15:36,280 INFO: Train Loss:0.735 | Acc:0.8495 | F1:0.1674
    2022-05-11 03:15:47,535 INFO: val Loss:0.626 | Acc:0.8551 | F1:0.1907
    2022-05-11 03:15:49,482 INFO: -----------------SAVE:15epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-11 03:17:49,170 INFO: Epoch:[016/200]
    2022-05-11 03:17:49,171 INFO: Train Loss:0.693 | Acc:0.8521 | F1:0.1815
    2022-05-11 03:18:00,268 INFO: val Loss:0.593 | Acc:0.8598 | F1:0.2074
    2022-05-11 03:18:02,324 INFO: -----------------SAVE:16epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-11 03:19:59,121 INFO: Epoch:[017/200]
    2022-05-11 03:19:59,121 INFO: Train Loss:0.645 | Acc:0.8579 | F1:0.2212
    2022-05-11 03:20:09,377 INFO: val Loss:0.535 | Acc:0.8680 | F1:0.2535
    2022-05-11 03:20:11,496 INFO: -----------------SAVE:17epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 03:22:06,946 INFO: Epoch:[018/200]
    2022-05-11 03:22:06,946 INFO: Train Loss:0.584 | Acc:0.8670 | F1:0.2635
    2022-05-11 03:22:16,758 INFO: val Loss:0.486 | Acc:0.8808 | F1:0.3234
    2022-05-11 03:22:18,640 INFO: -----------------SAVE:18epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:51<00:00,  1.92it/s]
    2022-05-11 03:24:10,228 INFO: Epoch:[019/200]
    2022-05-11 03:24:10,228 INFO: Train Loss:0.530 | Acc:0.8781 | F1:0.3243
    2022-05-11 03:24:20,124 INFO: val Loss:0.446 | Acc:0.8879 | F1:0.3507
    2022-05-11 03:24:21,928 INFO: -----------------SAVE:19epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:51<00:00,  1.93it/s]
    2022-05-11 03:26:13,073 INFO: Epoch:[020/200]
    2022-05-11 03:26:13,073 INFO: Train Loss:0.501 | Acc:0.8831 | F1:0.3728
    2022-05-11 03:26:22,928 INFO: val Loss:0.402 | Acc:0.9019 | F1:0.4279
    2022-05-11 03:26:24,736 INFO: -----------------SAVE:20epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:51<00:00,  1.92it/s]
    2022-05-11 03:28:15,937 INFO: Epoch:[021/200]
    2022-05-11 03:28:15,937 INFO: Train Loss:0.461 | Acc:0.8942 | F1:0.4279
    2022-05-11 03:28:25,757 INFO: val Loss:0.357 | Acc:0.9112 | F1:0.4765
    2022-05-11 03:28:27,587 INFO: -----------------SAVE:21epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:51<00:00,  1.92it/s]
    2022-05-11 03:30:19,115 INFO: Epoch:[022/200]
    2022-05-11 03:30:19,116 INFO: Train Loss:0.413 | Acc:0.8983 | F1:0.4531
    2022-05-11 03:30:28,972 INFO: val Loss:0.352 | Acc:0.9136 | F1:0.5100
    2022-05-11 03:30:30,806 INFO: -----------------SAVE:22epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:51<00:00,  1.92it/s]
    2022-05-11 03:32:22,019 INFO: Epoch:[023/200]
    2022-05-11 03:32:22,020 INFO: Train Loss:0.381 | Acc:0.9053 | F1:0.5046
    2022-05-11 03:32:31,800 INFO: val Loss:0.337 | Acc:0.9182 | F1:0.5447
    2022-05-11 03:32:33,628 INFO: -----------------SAVE:23epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.93it/s]
    2022-05-11 03:34:24,610 INFO: Epoch:[024/200]
    2022-05-11 03:34:24,611 INFO: Train Loss:0.372 | Acc:0.9047 | F1:0.5143
    2022-05-11 03:34:34,387 INFO: val Loss:0.278 | Acc:0.9182 | F1:0.5210
    2022-05-11 03:34:36,192 INFO: -----------------SAVE:24epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.93it/s]
    2022-05-11 03:36:26,900 INFO: Epoch:[025/200]
    2022-05-11 03:36:26,900 INFO: Train Loss:0.341 | Acc:0.9135 | F1:0.5586
    2022-05-11 03:36:36,640 INFO: val Loss:0.285 | Acc:0.9217 | F1:0.5612
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 03:38:27,235 INFO: Epoch:[026/200]
    2022-05-11 03:38:27,235 INFO: Train Loss:0.330 | Acc:0.9167 | F1:0.5779
    2022-05-11 03:38:37,025 INFO: val Loss:0.259 | Acc:0.9276 | F1:0.6013
    2022-05-11 03:38:39,018 INFO: -----------------SAVE:26epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.93it/s]
    2022-05-11 03:40:29,698 INFO: Epoch:[027/200]
    2022-05-11 03:40:29,698 INFO: Train Loss:0.305 | Acc:0.9214 | F1:0.5908
    2022-05-11 03:40:39,461 INFO: val Loss:0.240 | Acc:0.9322 | F1:0.6208
    2022-05-11 03:40:41,473 INFO: -----------------SAVE:27epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 03:42:31,999 INFO: Epoch:[028/200]
    2022-05-11 03:42:31,999 INFO: Train Loss:0.285 | Acc:0.9281 | F1:0.6469
    2022-05-11 03:42:41,778 INFO: val Loss:0.234 | Acc:0.9299 | F1:0.6064
    2022-05-11 03:42:43,503 INFO: -----------------SAVE:28epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.78it/s]
    2022-05-11 03:44:43,434 INFO: Epoch:[029/200]
    2022-05-11 03:44:43,435 INFO: Train Loss:0.279 | Acc:0.9252 | F1:0.6252
    2022-05-11 03:44:56,248 INFO: val Loss:0.234 | Acc:0.9346 | F1:0.6632
    2022-05-11 03:44:58,136 INFO: -----------------SAVE:29epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:54<00:00,  1.87it/s]
    2022-05-11 03:46:52,543 INFO: Epoch:[030/200]
    2022-05-11 03:46:52,544 INFO: Train Loss:0.270 | Acc:0.9281 | F1:0.6590
    2022-05-11 03:47:02,344 INFO: val Loss:0.247 | Acc:0.9311 | F1:0.6016
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 03:48:52,444 INFO: Epoch:[031/200]
    2022-05-11 03:48:52,444 INFO: Train Loss:0.277 | Acc:0.9313 | F1:0.6669
    2022-05-11 03:49:02,216 INFO: val Loss:0.225 | Acc:0.9439 | F1:0.6981
    2022-05-11 03:49:03,994 INFO: -----------------SAVE:31epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 03:50:54,584 INFO: Epoch:[032/200]
    2022-05-11 03:50:54,585 INFO: Train Loss:0.258 | Acc:0.9304 | F1:0.6795
    2022-05-11 03:51:04,332 INFO: val Loss:0.185 | Acc:0.9486 | F1:0.7218
    2022-05-11 03:51:06,087 INFO: -----------------SAVE:32epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.93it/s]
    2022-05-11 03:52:56,741 INFO: Epoch:[033/200]
    2022-05-11 03:52:56,742 INFO: Train Loss:0.234 | Acc:0.9392 | F1:0.7144
    2022-05-11 03:53:06,498 INFO: val Loss:0.229 | Acc:0.9463 | F1:0.7071
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 03:54:56,997 INFO: Epoch:[034/200]
    2022-05-11 03:54:56,998 INFO: Train Loss:0.232 | Acc:0.9374 | F1:0.7057
    2022-05-11 03:55:06,804 INFO: val Loss:0.193 | Acc:0.9451 | F1:0.7141
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 03:56:57,192 INFO: Epoch:[035/200]
    2022-05-11 03:56:57,192 INFO: Train Loss:0.240 | Acc:0.9363 | F1:0.7171
    2022-05-11 03:57:06,970 INFO: val Loss:0.202 | Acc:0.9451 | F1:0.7022
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 03:58:57,163 INFO: Epoch:[036/200]
    2022-05-11 03:58:57,164 INFO: Train Loss:0.225 | Acc:0.9404 | F1:0.7302
    2022-05-11 03:59:06,942 INFO: val Loss:0.232 | Acc:0.9369 | F1:0.7037
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 04:00:57,456 INFO: Epoch:[037/200]
    2022-05-11 04:00:57,457 INFO: Train Loss:0.213 | Acc:0.9448 | F1:0.7442
    2022-05-11 04:01:07,273 INFO: val Loss:0.221 | Acc:0.9474 | F1:0.7328
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 04:02:57,444 INFO: Epoch:[038/200]
    2022-05-11 04:02:57,444 INFO: Train Loss:0.226 | Acc:0.9383 | F1:0.7296
    2022-05-11 04:03:07,243 INFO: val Loss:0.232 | Acc:0.9241 | F1:0.7241
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 04:04:57,608 INFO: Epoch:[039/200]
    2022-05-11 04:04:57,608 INFO: Train Loss:0.203 | Acc:0.9471 | F1:0.7676
    2022-05-11 04:05:07,379 INFO: val Loss:0.199 | Acc:0.9498 | F1:0.7713
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 04:06:57,789 INFO: Epoch:[040/200]
    2022-05-11 04:06:57,790 INFO: Train Loss:0.218 | Acc:0.9445 | F1:0.7715
    2022-05-11 04:07:07,574 INFO: val Loss:0.223 | Acc:0.9381 | F1:0.6950
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 04:08:58,049 INFO: Epoch:[041/200]
    2022-05-11 04:08:58,049 INFO: Train Loss:0.195 | Acc:0.9459 | F1:0.7709
    2022-05-11 04:09:07,857 INFO: val Loss:0.242 | Acc:0.9346 | F1:0.6636
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.93it/s]
    2022-05-11 04:10:58,465 INFO: Epoch:[042/200]
    2022-05-11 04:10:58,466 INFO: Train Loss:0.188 | Acc:0.9486 | F1:0.7729
    2022-05-11 04:11:08,285 INFO: val Loss:0.273 | Acc:0.9287 | F1:0.7049
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:52<00:00,  1.90it/s]
    2022-05-11 04:13:00,738 INFO: Epoch:[043/200]
    2022-05-11 04:13:00,739 INFO: Train Loss:0.209 | Acc:0.9410 | F1:0.7514
    2022-05-11 04:13:10,594 INFO: val Loss:0.207 | Acc:0.9474 | F1:0.7358
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 04:15:03,824 INFO: Epoch:[044/200]
    2022-05-11 04:15:03,824 INFO: Train Loss:0.185 | Acc:0.9497 | F1:0.7901
    2022-05-11 04:15:13,699 INFO: val Loss:0.180 | Acc:0.9463 | F1:0.7528
    2022-05-11 04:15:15,557 INFO: -----------------SAVE:44epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:52<00:00,  1.91it/s]
    2022-05-11 04:17:07,651 INFO: Epoch:[045/200]
    2022-05-11 04:17:07,651 INFO: Train Loss:0.181 | Acc:0.9529 | F1:0.7957
    2022-05-11 04:17:17,431 INFO: val Loss:0.202 | Acc:0.9474 | F1:0.7228
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 04:19:07,650 INFO: Epoch:[046/200]
    2022-05-11 04:19:07,650 INFO: Train Loss:0.189 | Acc:0.9506 | F1:0.7772
    2022-05-11 04:19:17,391 INFO: val Loss:0.204 | Acc:0.9474 | F1:0.7411
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.93it/s]
    2022-05-11 04:21:08,212 INFO: Epoch:[047/200]
    2022-05-11 04:21:08,212 INFO: Train Loss:0.197 | Acc:0.9503 | F1:0.7999
    2022-05-11 04:21:17,998 INFO: val Loss:0.166 | Acc:0.9533 | F1:0.7719
    2022-05-11 04:21:19,775 INFO: -----------------SAVE:47epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.93it/s]
    2022-05-11 04:23:10,603 INFO: Epoch:[048/200]
    2022-05-11 04:23:10,603 INFO: Train Loss:0.203 | Acc:0.9483 | F1:0.7829
    2022-05-11 04:23:20,377 INFO: val Loss:0.195 | Acc:0.9474 | F1:0.7508
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 04:25:10,923 INFO: Epoch:[049/200]
    2022-05-11 04:25:10,924 INFO: Train Loss:0.182 | Acc:0.9491 | F1:0.7889
    2022-05-11 04:25:20,656 INFO: val Loss:0.191 | Acc:0.9498 | F1:0.7241
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.93it/s]
    2022-05-11 04:27:11,406 INFO: Epoch:[050/200]
    2022-05-11 04:27:11,406 INFO: Train Loss:0.189 | Acc:0.9459 | F1:0.7658
    2022-05-11 04:27:21,543 INFO: val Loss:0.207 | Acc:0.9451 | F1:0.7924
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 04:29:12,135 INFO: Epoch:[051/200]
    2022-05-11 04:29:12,135 INFO: Train Loss:0.194 | Acc:0.9494 | F1:0.7815
    2022-05-11 04:29:21,885 INFO: val Loss:0.198 | Acc:0.9428 | F1:0.7215
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:51<00:00,  1.93it/s]
    2022-05-11 04:31:13,011 INFO: Epoch:[052/200]
    2022-05-11 04:31:13,011 INFO: Train Loss:0.185 | Acc:0.9559 | F1:0.8231
    2022-05-11 04:31:22,800 INFO: val Loss:0.216 | Acc:0.9568 | F1:0.7729
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.93it/s]
    2022-05-11 04:33:13,422 INFO: Epoch:[053/200]
    2022-05-11 04:33:13,423 INFO: Train Loss:0.181 | Acc:0.9562 | F1:0.8121
    2022-05-11 04:33:23,242 INFO: val Loss:0.193 | Acc:0.9544 | F1:0.7588
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:52<00:00,  1.91it/s]
    2022-05-11 04:35:15,253 INFO: Epoch:[054/200]
    2022-05-11 04:35:15,253 INFO: Train Loss:0.175 | Acc:0.9532 | F1:0.7955
    2022-05-11 04:35:25,033 INFO: val Loss:0.291 | Acc:0.9428 | F1:0.7564
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:51<00:00,  1.93it/s]
    2022-05-11 04:37:16,169 INFO: Epoch:[055/200]
    2022-05-11 04:37:16,169 INFO: Train Loss:0.177 | Acc:0.9541 | F1:0.8002
    2022-05-11 04:37:25,936 INFO: val Loss:0.250 | Acc:0.9287 | F1:0.7617
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.93it/s]
    2022-05-11 04:39:16,814 INFO: Epoch:[056/200]
    2022-05-11 04:39:16,815 INFO: Train Loss:0.175 | Acc:0.9529 | F1:0.7998
    2022-05-11 04:39:26,592 INFO: val Loss:0.167 | Acc:0.9603 | F1:0.7941
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:51<00:00,  1.92it/s]
    2022-05-11 04:41:18,339 INFO: Epoch:[057/200]
    2022-05-11 04:41:18,340 INFO: Train Loss:0.171 | Acc:0.9556 | F1:0.8264
    2022-05-11 04:41:28,087 INFO: val Loss:0.207 | Acc:0.9439 | F1:0.7398
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 04:43:18,650 INFO: Epoch:[058/200]
    2022-05-11 04:43:18,651 INFO: Train Loss:0.154 | Acc:0.9602 | F1:0.8402
    2022-05-11 04:43:28,407 INFO: val Loss:0.209 | Acc:0.9474 | F1:0.6904
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.93it/s]
    2022-05-11 04:45:19,159 INFO: Epoch:[059/200]
    2022-05-11 04:45:19,159 INFO: Train Loss:0.151 | Acc:0.9585 | F1:0.8444
    2022-05-11 04:45:28,939 INFO: val Loss:0.202 | Acc:0.9521 | F1:0.7654
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 04:47:19,387 INFO: Epoch:[060/200]
    2022-05-11 04:47:19,387 INFO: Train Loss:0.164 | Acc:0.9564 | F1:0.8267
    2022-05-11 04:47:29,134 INFO: val Loss:0.185 | Acc:0.9498 | F1:0.7491
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.93it/s]
    2022-05-11 04:49:19,980 INFO: Epoch:[061/200]
    2022-05-11 04:49:19,981 INFO: Train Loss:0.159 | Acc:0.9562 | F1:0.8176
    2022-05-11 04:49:29,826 INFO: val Loss:0.196 | Acc:0.9556 | F1:0.7943
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.93it/s]
    2022-05-11 04:51:20,729 INFO: Epoch:[062/200]
    2022-05-11 04:51:20,730 INFO: Train Loss:0.169 | Acc:0.9570 | F1:0.8242
    2022-05-11 04:51:30,508 INFO: val Loss:0.217 | Acc:0.9346 | F1:0.7760
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.93it/s]
    2022-05-11 04:53:21,337 INFO: Epoch:[063/200]
    2022-05-11 04:53:21,337 INFO: Train Loss:0.146 | Acc:0.9594 | F1:0.8438
    2022-05-11 04:53:31,087 INFO: val Loss:0.170 | Acc:0.9521 | F1:0.7472
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:51<00:00,  1.93it/s]
    2022-05-11 04:55:22,226 INFO: Epoch:[064/200]
    2022-05-11 04:55:22,227 INFO: Train Loss:0.145 | Acc:0.9643 | F1:0.8514
    2022-05-11 04:55:31,966 INFO: val Loss:0.172 | Acc:0.9544 | F1:0.8041
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 04:57:22,459 INFO: Epoch:[065/200]
    2022-05-11 04:57:22,459 INFO: Train Loss:0.161 | Acc:0.9567 | F1:0.8323
    2022-05-11 04:57:32,110 INFO: val Loss:0.171 | Acc:0.9533 | F1:0.7592
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 04:59:21,776 INFO: Epoch:[066/200]
    2022-05-11 04:59:21,776 INFO: Train Loss:0.147 | Acc:0.9588 | F1:0.8272
    2022-05-11 04:59:31,435 INFO: val Loss:0.292 | Acc:0.9007 | F1:0.7094
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:01:21,103 INFO: Epoch:[067/200]
    2022-05-11 05:01:21,103 INFO: Train Loss:0.155 | Acc:0.9623 | F1:0.8439
    2022-05-11 05:01:30,760 INFO: val Loss:0.163 | Acc:0.9568 | F1:0.7962
    2022-05-11 05:01:32,536 INFO: -----------------SAVE:67epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:03:22,368 INFO: Epoch:[068/200]
    2022-05-11 05:03:22,368 INFO: Train Loss:0.130 | Acc:0.9608 | F1:0.8469
    2022-05-11 05:03:32,020 INFO: val Loss:0.186 | Acc:0.9556 | F1:0.7821
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:05:21,739 INFO: Epoch:[069/200]
    2022-05-11 05:05:21,739 INFO: Train Loss:0.137 | Acc:0.9635 | F1:0.8553
    2022-05-11 05:05:31,401 INFO: val Loss:0.210 | Acc:0.9486 | F1:0.7735
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:07:21,055 INFO: Epoch:[070/200]
    2022-05-11 05:07:21,056 INFO: Train Loss:0.138 | Acc:0.9617 | F1:0.8572
    2022-05-11 05:07:30,703 INFO: val Loss:0.245 | Acc:0.9498 | F1:0.7489
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:09:20,302 INFO: Epoch:[071/200]
    2022-05-11 05:09:20,302 INFO: Train Loss:0.133 | Acc:0.9649 | F1:0.8622
    2022-05-11 05:09:29,959 INFO: val Loss:0.203 | Acc:0.9521 | F1:0.7338
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:11:19,499 INFO: Epoch:[072/200]
    2022-05-11 05:11:19,500 INFO: Train Loss:0.137 | Acc:0.9640 | F1:0.8658
    2022-05-11 05:11:29,164 INFO: val Loss:0.204 | Acc:0.9556 | F1:0.7975
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:13:18,999 INFO: Epoch:[073/200]
    2022-05-11 05:13:18,999 INFO: Train Loss:0.140 | Acc:0.9600 | F1:0.8434
    2022-05-11 05:13:28,648 INFO: val Loss:0.188 | Acc:0.9556 | F1:0.7767
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:15:18,183 INFO: Epoch:[074/200]
    2022-05-11 05:15:18,183 INFO: Train Loss:0.117 | Acc:0.9646 | F1:0.8583
    2022-05-11 05:15:27,850 INFO: val Loss:0.213 | Acc:0.9498 | F1:0.7897
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.96it/s]
    2022-05-11 05:17:17,293 INFO: Epoch:[075/200]
    2022-05-11 05:17:17,293 INFO: Train Loss:0.132 | Acc:0.9652 | F1:0.8515
    2022-05-11 05:17:26,928 INFO: val Loss:0.154 | Acc:0.9591 | F1:0.8379
    2022-05-11 05:17:28,662 INFO: -----------------SAVE:75epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:19:18,542 INFO: Epoch:[076/200]
    2022-05-11 05:19:18,542 INFO: Train Loss:0.142 | Acc:0.9632 | F1:0.8634
    2022-05-11 05:19:28,200 INFO: val Loss:0.226 | Acc:0.9474 | F1:0.7456
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.96it/s]
    2022-05-11 05:21:17,642 INFO: Epoch:[077/200]
    2022-05-11 05:21:17,642 INFO: Train Loss:0.112 | Acc:0.9678 | F1:0.8709
    2022-05-11 05:21:27,287 INFO: val Loss:0.173 | Acc:0.9603 | F1:0.8149
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:23:16,949 INFO: Epoch:[078/200]
    2022-05-11 05:23:16,949 INFO: Train Loss:0.115 | Acc:0.9684 | F1:0.8809
    2022-05-11 05:23:26,580 INFO: val Loss:0.197 | Acc:0.9474 | F1:0.7616
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:25:16,479 INFO: Epoch:[079/200]
    2022-05-11 05:25:16,480 INFO: Train Loss:0.130 | Acc:0.9673 | F1:0.8803
    2022-05-11 05:25:26,137 INFO: val Loss:0.167 | Acc:0.9591 | F1:0.8054
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:27:15,850 INFO: Epoch:[080/200]
    2022-05-11 05:27:15,851 INFO: Train Loss:0.130 | Acc:0.9658 | F1:0.8691
    2022-05-11 05:27:25,537 INFO: val Loss:0.141 | Acc:0.9626 | F1:0.8030
    2022-05-11 05:27:27,270 INFO: -----------------SAVE:80epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:29:16,963 INFO: Epoch:[081/200]
    2022-05-11 05:29:16,963 INFO: Train Loss:0.128 | Acc:0.9652 | F1:0.8577
    2022-05-11 05:29:26,621 INFO: val Loss:0.146 | Acc:0.9650 | F1:0.8300
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:31:16,327 INFO: Epoch:[082/200]
    2022-05-11 05:31:16,327 INFO: Train Loss:0.120 | Acc:0.9684 | F1:0.8737
    2022-05-11 05:31:25,965 INFO: val Loss:0.187 | Acc:0.9533 | F1:0.7669
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:33:15,710 INFO: Epoch:[083/200]
    2022-05-11 05:33:15,710 INFO: Train Loss:0.129 | Acc:0.9690 | F1:0.8798
    2022-05-11 05:33:25,376 INFO: val Loss:0.146 | Acc:0.9603 | F1:0.8155
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:35:15,200 INFO: Epoch:[084/200]
    2022-05-11 05:35:15,200 INFO: Train Loss:0.112 | Acc:0.9711 | F1:0.8903
    2022-05-11 05:35:24,839 INFO: val Loss:0.118 | Acc:0.9661 | F1:0.8385
    2022-05-11 05:35:26,707 INFO: -----------------SAVE:84epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:37:16,337 INFO: Epoch:[085/200]
    2022-05-11 05:37:16,337 INFO: Train Loss:0.124 | Acc:0.9652 | F1:0.8758
    2022-05-11 05:37:26,009 INFO: val Loss:0.257 | Acc:0.9276 | F1:0.7217
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:39:15,625 INFO: Epoch:[086/200]
    2022-05-11 05:39:15,625 INFO: Train Loss:0.120 | Acc:0.9673 | F1:0.8727
    2022-05-11 05:39:25,282 INFO: val Loss:0.200 | Acc:0.9451 | F1:0.7470
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:41:15,186 INFO: Epoch:[087/200]
    2022-05-11 05:41:15,186 INFO: Train Loss:0.120 | Acc:0.9667 | F1:0.8779
    2022-05-11 05:41:24,845 INFO: val Loss:0.154 | Acc:0.9614 | F1:0.8409
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:43:14,552 INFO: Epoch:[088/200]
    2022-05-11 05:43:14,552 INFO: Train Loss:0.095 | Acc:0.9740 | F1:0.9004
    2022-05-11 05:43:24,193 INFO: val Loss:0.166 | Acc:0.9591 | F1:0.7914
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:45:13,881 INFO: Epoch:[089/200]
    2022-05-11 05:45:13,881 INFO: Train Loss:0.121 | Acc:0.9716 | F1:0.8878
    2022-05-11 05:45:23,710 INFO: val Loss:0.173 | Acc:0.9579 | F1:0.7942
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 05:47:14,198 INFO: Epoch:[090/200]
    2022-05-11 05:47:14,198 INFO: Train Loss:0.111 | Acc:0.9708 | F1:0.8923
    2022-05-11 05:47:23,855 INFO: val Loss:0.141 | Acc:0.9638 | F1:0.8251
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:49:13,589 INFO: Epoch:[091/200]
    2022-05-11 05:49:13,589 INFO: Train Loss:0.116 | Acc:0.9708 | F1:0.8903
    2022-05-11 05:49:23,280 INFO: val Loss:0.173 | Acc:0.9591 | F1:0.8178
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:51:13,004 INFO: Epoch:[092/200]
    2022-05-11 05:51:13,004 INFO: Train Loss:0.107 | Acc:0.9734 | F1:0.9027
    2022-05-11 05:51:22,677 INFO: val Loss:0.148 | Acc:0.9661 | F1:0.8014
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:53:12,552 INFO: Epoch:[093/200]
    2022-05-11 05:53:12,553 INFO: Train Loss:0.099 | Acc:0.9705 | F1:0.8864
    2022-05-11 05:53:22,207 INFO: val Loss:0.149 | Acc:0.9614 | F1:0.7953
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:55:11,939 INFO: Epoch:[094/200]
    2022-05-11 05:55:11,940 INFO: Train Loss:0.097 | Acc:0.9757 | F1:0.8989
    2022-05-11 05:55:21,616 INFO: val Loss:0.161 | Acc:0.9521 | F1:0.7909
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:57:11,294 INFO: Epoch:[095/200]
    2022-05-11 05:57:11,295 INFO: Train Loss:0.115 | Acc:0.9719 | F1:0.8984
    2022-05-11 05:57:20,962 INFO: val Loss:0.148 | Acc:0.9638 | F1:0.8240
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 05:59:10,552 INFO: Epoch:[096/200]
    2022-05-11 05:59:10,552 INFO: Train Loss:0.112 | Acc:0.9714 | F1:0.8985
    2022-05-11 05:59:20,237 INFO: val Loss:0.194 | Acc:0.9626 | F1:0.8044
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 06:01:10,277 INFO: Epoch:[097/200]
    2022-05-11 06:01:10,277 INFO: Train Loss:0.097 | Acc:0.9740 | F1:0.9000
    2022-05-11 06:01:20,033 INFO: val Loss:0.332 | Acc:0.9276 | F1:0.7832
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:52<00:00,  1.91it/s]
    2022-05-11 06:03:12,219 INFO: Epoch:[098/200]
    2022-05-11 06:03:12,219 INFO: Train Loss:0.097 | Acc:0.9714 | F1:0.8952
    2022-05-11 06:03:21,958 INFO: val Loss:0.169 | Acc:0.9603 | F1:0.8095
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:51<00:00,  1.91it/s]
    2022-05-11 06:05:13,775 INFO: Epoch:[099/200]
    2022-05-11 06:05:13,775 INFO: Train Loss:0.102 | Acc:0.9708 | F1:0.9015
    2022-05-11 06:05:23,531 INFO: val Loss:0.176 | Acc:0.9638 | F1:0.8480
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:51<00:00,  1.91it/s]
    2022-05-11 06:07:15,329 INFO: Epoch:[100/200]
    2022-05-11 06:07:15,330 INFO: Train Loss:0.093 | Acc:0.9769 | F1:0.9136
    2022-05-11 06:07:25,073 INFO: val Loss:0.194 | Acc:0.9568 | F1:0.7671
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 06:09:18,725 INFO: Epoch:[101/200]
    2022-05-11 06:09:18,725 INFO: Train Loss:0.086 | Acc:0.9772 | F1:0.9157
    2022-05-11 06:09:28,692 INFO: val Loss:0.218 | Acc:0.9579 | F1:0.7801
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:52<00:00,  1.90it/s]
    2022-05-11 06:11:21,571 INFO: Epoch:[102/200]
    2022-05-11 06:11:21,571 INFO: Train Loss:0.100 | Acc:0.9722 | F1:0.8909
    2022-05-11 06:11:31,373 INFO: val Loss:0.236 | Acc:0.9299 | F1:0.7349
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:52<00:00,  1.90it/s]
    2022-05-11 06:13:23,732 INFO: Epoch:[103/200]
    2022-05-11 06:13:23,733 INFO: Train Loss:0.100 | Acc:0.9728 | F1:0.8953
    2022-05-11 06:13:33,527 INFO: val Loss:0.175 | Acc:0.9556 | F1:0.7806
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:52<00:00,  1.90it/s]
    2022-05-11 06:15:26,087 INFO: Epoch:[104/200]
    2022-05-11 06:15:26,087 INFO: Train Loss:0.092 | Acc:0.9737 | F1:0.9052
    2022-05-11 06:15:35,909 INFO: val Loss:0.217 | Acc:0.9533 | F1:0.7803
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:51<00:00,  1.91it/s]
    2022-05-11 06:17:27,904 INFO: Epoch:[105/200]
    2022-05-11 06:17:27,904 INFO: Train Loss:0.093 | Acc:0.9754 | F1:0.9176
    2022-05-11 06:17:37,678 INFO: val Loss:0.213 | Acc:0.9591 | F1:0.8115
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:52<00:00,  1.90it/s]
    2022-05-11 06:19:30,127 INFO: Epoch:[106/200]
    2022-05-11 06:19:30,127 INFO: Train Loss:0.079 | Acc:0.9766 | F1:0.9155
    2022-05-11 06:19:39,882 INFO: val Loss:0.175 | Acc:0.9638 | F1:0.8123
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 06:21:29,802 INFO: Epoch:[107/200]
    2022-05-11 06:21:29,802 INFO: Train Loss:0.095 | Acc:0.9772 | F1:0.9127
    2022-05-11 06:21:39,487 INFO: val Loss:0.166 | Acc:0.9591 | F1:0.7968
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 06:23:29,224 INFO: Epoch:[108/200]
    2022-05-11 06:23:29,224 INFO: Train Loss:0.077 | Acc:0.9810 | F1:0.9358
    2022-05-11 06:23:38,878 INFO: val Loss:0.165 | Acc:0.9603 | F1:0.8071
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 06:25:28,701 INFO: Epoch:[109/200]
    2022-05-11 06:25:28,701 INFO: Train Loss:0.093 | Acc:0.9766 | F1:0.9200
    2022-05-11 06:25:38,337 INFO: val Loss:0.166 | Acc:0.9626 | F1:0.7990
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 06:27:28,503 INFO: Epoch:[110/200]
    2022-05-11 06:27:28,503 INFO: Train Loss:0.071 | Acc:0.9804 | F1:0.9286
    2022-05-11 06:27:38,170 INFO: val Loss:0.187 | Acc:0.9533 | F1:0.7835
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 06:29:27,982 INFO: Epoch:[111/200]
    2022-05-11 06:29:27,982 INFO: Train Loss:0.072 | Acc:0.9798 | F1:0.9294
    2022-05-11 06:29:37,633 INFO: val Loss:0.192 | Acc:0.9579 | F1:0.7949
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 06:31:27,412 INFO: Epoch:[112/200]
    2022-05-11 06:31:27,412 INFO: Train Loss:0.095 | Acc:0.9763 | F1:0.9077
    2022-05-11 06:31:37,086 INFO: val Loss:0.181 | Acc:0.9614 | F1:0.8139
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 06:33:26,723 INFO: Epoch:[113/200]
    2022-05-11 06:33:26,723 INFO: Train Loss:0.085 | Acc:0.9763 | F1:0.9151
    2022-05-11 06:33:36,422 INFO: val Loss:0.142 | Acc:0.9638 | F1:0.8159
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 06:35:26,403 INFO: Epoch:[114/200]
    2022-05-11 06:35:26,404 INFO: Train Loss:0.096 | Acc:0.9772 | F1:0.9140
    2022-05-11 06:35:36,097 INFO: val Loss:0.151 | Acc:0.9673 | F1:0.8336
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 06:37:26,031 INFO: Epoch:[115/200]
    2022-05-11 06:37:26,032 INFO: Train Loss:0.081 | Acc:0.9795 | F1:0.9296
    2022-05-11 06:37:35,724 INFO: val Loss:0.168 | Acc:0.9650 | F1:0.8299
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 06:39:25,769 INFO: Epoch:[116/200]
    2022-05-11 06:39:25,770 INFO: Train Loss:0.062 | Acc:0.9828 | F1:0.9450
    2022-05-11 06:39:35,419 INFO: val Loss:0.136 | Acc:0.9673 | F1:0.8219
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 06:41:25,170 INFO: Epoch:[117/200]
    2022-05-11 06:41:25,170 INFO: Train Loss:0.075 | Acc:0.9787 | F1:0.9231
    2022-05-11 06:41:34,860 INFO: val Loss:0.215 | Acc:0.9474 | F1:0.7915
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 06:43:24,714 INFO: Epoch:[118/200]
    2022-05-11 06:43:24,714 INFO: Train Loss:0.073 | Acc:0.9819 | F1:0.9361
    2022-05-11 06:43:34,333 INFO: val Loss:0.176 | Acc:0.9626 | F1:0.8370
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 06:45:24,244 INFO: Epoch:[119/200]
    2022-05-11 06:45:24,245 INFO: Train Loss:0.081 | Acc:0.9778 | F1:0.9208
    2022-05-11 06:45:33,922 INFO: val Loss:0.186 | Acc:0.9521 | F1:0.7738
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 06:47:23,684 INFO: Epoch:[120/200]
    2022-05-11 06:47:23,684 INFO: Train Loss:0.083 | Acc:0.9763 | F1:0.9131
    2022-05-11 06:47:33,394 INFO: val Loss:0.198 | Acc:0.9533 | F1:0.8196
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:52<00:00,  1.91it/s]
    2022-05-11 06:49:25,455 INFO: Epoch:[121/200]
    2022-05-11 06:49:25,455 INFO: Train Loss:0.070 | Acc:0.9828 | F1:0.9411
    2022-05-11 06:49:35,251 INFO: val Loss:0.209 | Acc:0.9544 | F1:0.8448
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:51<00:00,  1.92it/s]
    2022-05-11 06:51:26,447 INFO: Epoch:[122/200]
    2022-05-11 06:51:26,447 INFO: Train Loss:0.079 | Acc:0.9792 | F1:0.9249
    2022-05-11 06:51:36,241 INFO: val Loss:0.143 | Acc:0.9556 | F1:0.7970
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:51<00:00,  1.93it/s]
    2022-05-11 06:53:27,314 INFO: Epoch:[123/200]
    2022-05-11 06:53:27,315 INFO: Train Loss:0.069 | Acc:0.9833 | F1:0.9415
    2022-05-11 06:53:37,089 INFO: val Loss:0.152 | Acc:0.9685 | F1:0.8429
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:51<00:00,  1.93it/s]
    2022-05-11 06:55:28,132 INFO: Epoch:[124/200]
    2022-05-11 06:55:28,133 INFO: Train Loss:0.061 | Acc:0.9830 | F1:0.9403
    2022-05-11 06:55:37,927 INFO: val Loss:0.183 | Acc:0.9603 | F1:0.8213
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:51<00:00,  1.93it/s]
    2022-05-11 06:57:29,010 INFO: Epoch:[125/200]
    2022-05-11 06:57:29,011 INFO: Train Loss:0.070 | Acc:0.9801 | F1:0.9273
    2022-05-11 06:57:38,765 INFO: val Loss:0.118 | Acc:0.9708 | F1:0.8478
    2022-05-11 06:57:40,517 INFO: -----------------SAVE:125epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.93it/s]
    2022-05-11 06:59:31,519 INFO: Epoch:[126/200]
    2022-05-11 06:59:31,520 INFO: Train Loss:0.059 | Acc:0.9833 | F1:0.9419
    2022-05-11 06:59:41,296 INFO: val Loss:0.150 | Acc:0.9708 | F1:0.8631
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.93it/s]
    2022-05-11 07:01:32,297 INFO: Epoch:[127/200]
    2022-05-11 07:01:32,297 INFO: Train Loss:0.062 | Acc:0.9830 | F1:0.9353
    2022-05-11 07:01:42,159 INFO: val Loss:0.156 | Acc:0.9685 | F1:0.8674
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 07:03:32,123 INFO: Epoch:[128/200]
    2022-05-11 07:03:32,123 INFO: Train Loss:0.060 | Acc:0.9851 | F1:0.9501
    2022-05-11 07:03:41,804 INFO: val Loss:0.173 | Acc:0.9720 | F1:0.8801
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 07:05:31,422 INFO: Epoch:[129/200]
    2022-05-11 07:05:31,422 INFO: Train Loss:0.088 | Acc:0.9769 | F1:0.9141
    2022-05-11 07:05:41,098 INFO: val Loss:0.127 | Acc:0.9673 | F1:0.8563
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 07:07:30,976 INFO: Epoch:[130/200]
    2022-05-11 07:07:30,977 INFO: Train Loss:0.066 | Acc:0.9801 | F1:0.9302
    2022-05-11 07:07:40,655 INFO: val Loss:0.157 | Acc:0.9661 | F1:0.8755
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 07:09:30,359 INFO: Epoch:[131/200]
    2022-05-11 07:09:30,359 INFO: Train Loss:0.058 | Acc:0.9845 | F1:0.9422
    2022-05-11 07:09:40,029 INFO: val Loss:0.151 | Acc:0.9696 | F1:0.8824
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 07:11:29,896 INFO: Epoch:[132/200]
    2022-05-11 07:11:29,896 INFO: Train Loss:0.047 | Acc:0.9877 | F1:0.9607
    2022-05-11 07:11:39,548 INFO: val Loss:0.154 | Acc:0.9673 | F1:0.8548
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 07:13:29,527 INFO: Epoch:[133/200]
    2022-05-11 07:13:29,527 INFO: Train Loss:0.058 | Acc:0.9839 | F1:0.9387
    2022-05-11 07:13:39,176 INFO: val Loss:0.145 | Acc:0.9685 | F1:0.8642
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 07:15:29,114 INFO: Epoch:[134/200]
    2022-05-11 07:15:29,114 INFO: Train Loss:0.058 | Acc:0.9833 | F1:0.9452
    2022-05-11 07:15:38,799 INFO: val Loss:0.142 | Acc:0.9696 | F1:0.8473
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 07:17:28,832 INFO: Epoch:[135/200]
    2022-05-11 07:17:28,832 INFO: Train Loss:0.052 | Acc:0.9866 | F1:0.9568
    2022-05-11 07:17:38,492 INFO: val Loss:0.159 | Acc:0.9661 | F1:0.8673
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.95it/s]
    2022-05-11 07:19:28,512 INFO: Epoch:[136/200]
    2022-05-11 07:19:28,512 INFO: Train Loss:0.051 | Acc:0.9842 | F1:0.9408
    2022-05-11 07:19:38,165 INFO: val Loss:0.146 | Acc:0.9708 | F1:0.8512
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 07:21:28,299 INFO: Epoch:[137/200]
    2022-05-11 07:21:28,300 INFO: Train Loss:0.060 | Acc:0.9845 | F1:0.9447
    2022-05-11 07:21:37,958 INFO: val Loss:0.121 | Acc:0.9731 | F1:0.8741
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 07:23:27,808 INFO: Epoch:[138/200]
    2022-05-11 07:23:27,809 INFO: Train Loss:0.053 | Acc:0.9845 | F1:0.9472
    2022-05-11 07:23:37,497 INFO: val Loss:0.185 | Acc:0.9650 | F1:0.8392
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 07:25:27,425 INFO: Epoch:[139/200]
    2022-05-11 07:25:27,425 INFO: Train Loss:0.051 | Acc:0.9854 | F1:0.9531
    2022-05-11 07:25:37,079 INFO: val Loss:0.198 | Acc:0.9544 | F1:0.8150
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 07:27:27,297 INFO: Epoch:[140/200]
    2022-05-11 07:27:27,298 INFO: Train Loss:0.052 | Acc:0.9851 | F1:0.9482
    2022-05-11 07:27:37,016 INFO: val Loss:0.174 | Acc:0.9568 | F1:0.7959
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 07:29:26,701 INFO: Epoch:[141/200]
    2022-05-11 07:29:26,701 INFO: Train Loss:0.046 | Acc:0.9889 | F1:0.9630
    2022-05-11 07:29:36,396 INFO: val Loss:0.159 | Acc:0.9626 | F1:0.8110
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 07:31:26,170 INFO: Epoch:[142/200]
    2022-05-11 07:31:26,171 INFO: Train Loss:0.048 | Acc:0.9886 | F1:0.9589
    2022-05-11 07:31:35,842 INFO: val Loss:0.170 | Acc:0.9603 | F1:0.8219
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 07:33:25,901 INFO: Epoch:[143/200]
    2022-05-11 07:33:25,901 INFO: Train Loss:0.046 | Acc:0.9889 | F1:0.9598
    2022-05-11 07:33:35,578 INFO: val Loss:0.161 | Acc:0.9673 | F1:0.8255
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 07:35:25,711 INFO: Epoch:[144/200]
    2022-05-11 07:35:25,711 INFO: Train Loss:0.043 | Acc:0.9871 | F1:0.9500
    2022-05-11 07:35:35,409 INFO: val Loss:0.175 | Acc:0.9614 | F1:0.7988
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 07:37:25,375 INFO: Epoch:[145/200]
    2022-05-11 07:37:25,376 INFO: Train Loss:0.051 | Acc:0.9839 | F1:0.9484
    2022-05-11 07:37:35,041 INFO: val Loss:0.149 | Acc:0.9731 | F1:0.8689
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 07:39:25,106 INFO: Epoch:[146/200]
    2022-05-11 07:39:25,107 INFO: Train Loss:0.048 | Acc:0.9860 | F1:0.9532
    2022-05-11 07:39:34,767 INFO: val Loss:0.189 | Acc:0.9568 | F1:0.8238
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 07:41:24,602 INFO: Epoch:[147/200]
    2022-05-11 07:41:24,603 INFO: Train Loss:0.043 | Acc:0.9883 | F1:0.9577
    2022-05-11 07:41:34,268 INFO: val Loss:0.162 | Acc:0.9661 | F1:0.8558
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 07:43:23,929 INFO: Epoch:[148/200]
    2022-05-11 07:43:23,929 INFO: Train Loss:0.052 | Acc:0.9868 | F1:0.9562
    2022-05-11 07:43:33,601 INFO: val Loss:0.141 | Acc:0.9708 | F1:0.8385
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 07:45:23,533 INFO: Epoch:[149/200]
    2022-05-11 07:45:23,533 INFO: Train Loss:0.041 | Acc:0.9901 | F1:0.9692
    2022-05-11 07:45:33,189 INFO: val Loss:0.155 | Acc:0.9731 | F1:0.8515
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 07:47:23,475 INFO: Epoch:[150/200]
    2022-05-11 07:47:23,476 INFO: Train Loss:0.037 | Acc:0.9906 | F1:0.9688
    2022-05-11 07:47:33,330 INFO: val Loss:0.164 | Acc:0.9685 | F1:0.8435
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 07:49:23,813 INFO: Epoch:[151/200]
    2022-05-11 07:49:23,813 INFO: Train Loss:0.041 | Acc:0.9895 | F1:0.9633
    2022-05-11 07:49:33,470 INFO: val Loss:0.152 | Acc:0.9731 | F1:0.8703
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 07:51:23,463 INFO: Epoch:[152/200]
    2022-05-11 07:51:23,463 INFO: Train Loss:0.037 | Acc:0.9895 | F1:0.9648
    2022-05-11 07:51:33,154 INFO: val Loss:0.174 | Acc:0.9626 | F1:0.8165
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 07:53:23,020 INFO: Epoch:[153/200]
    2022-05-11 07:53:23,020 INFO: Train Loss:0.040 | Acc:0.9880 | F1:0.9618
    2022-05-11 07:53:32,670 INFO: val Loss:0.157 | Acc:0.9685 | F1:0.8580
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 07:55:22,604 INFO: Epoch:[154/200]
    2022-05-11 07:55:22,604 INFO: Train Loss:0.033 | Acc:0.9889 | F1:0.9616
    2022-05-11 07:55:32,300 INFO: val Loss:0.169 | Acc:0.9661 | F1:0.8220
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 07:57:22,334 INFO: Epoch:[155/200]
    2022-05-11 07:57:22,334 INFO: Train Loss:0.025 | Acc:0.9927 | F1:0.9750
    2022-05-11 07:57:31,992 INFO: val Loss:0.149 | Acc:0.9696 | F1:0.8477
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.95it/s]
    2022-05-11 07:59:22,013 INFO: Epoch:[156/200]
    2022-05-11 07:59:22,013 INFO: Train Loss:0.037 | Acc:0.9912 | F1:0.9719
    2022-05-11 07:59:31,729 INFO: val Loss:0.148 | Acc:0.9696 | F1:0.8383
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 08:01:21,466 INFO: Epoch:[157/200]
    2022-05-11 08:01:21,466 INFO: Train Loss:0.030 | Acc:0.9918 | F1:0.9675
    2022-05-11 08:01:31,122 INFO: val Loss:0.164 | Acc:0.9626 | F1:0.8310
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 08:03:21,236 INFO: Epoch:[158/200]
    2022-05-11 08:03:21,236 INFO: Train Loss:0.033 | Acc:0.9909 | F1:0.9659
    2022-05-11 08:03:30,914 INFO: val Loss:0.174 | Acc:0.9614 | F1:0.8215
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 08:05:20,969 INFO: Epoch:[159/200]
    2022-05-11 08:05:20,970 INFO: Train Loss:0.036 | Acc:0.9915 | F1:0.9760
    2022-05-11 08:05:30,654 INFO: val Loss:0.163 | Acc:0.9685 | F1:0.8464
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 08:07:20,321 INFO: Epoch:[160/200]
    2022-05-11 08:07:20,322 INFO: Train Loss:0.036 | Acc:0.9898 | F1:0.9676
    2022-05-11 08:07:29,998 INFO: val Loss:0.145 | Acc:0.9638 | F1:0.8221
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 08:09:20,216 INFO: Epoch:[161/200]
    2022-05-11 08:09:20,217 INFO: Train Loss:0.026 | Acc:0.9933 | F1:0.9757
    2022-05-11 08:09:29,913 INFO: val Loss:0.142 | Acc:0.9685 | F1:0.8471
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 08:11:19,903 INFO: Epoch:[162/200]
    2022-05-11 08:11:19,904 INFO: Train Loss:0.024 | Acc:0.9927 | F1:0.9776
    2022-05-11 08:11:29,589 INFO: val Loss:0.170 | Acc:0.9696 | F1:0.8535
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 08:13:19,870 INFO: Epoch:[163/200]
    2022-05-11 08:13:19,871 INFO: Train Loss:0.027 | Acc:0.9927 | F1:0.9754
    2022-05-11 08:13:29,538 INFO: val Loss:0.141 | Acc:0.9731 | F1:0.8646
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 08:15:19,614 INFO: Epoch:[164/200]
    2022-05-11 08:15:19,614 INFO: Train Loss:0.029 | Acc:0.9909 | F1:0.9699
    2022-05-11 08:15:29,319 INFO: val Loss:0.154 | Acc:0.9743 | F1:0.8783
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 08:17:19,258 INFO: Epoch:[165/200]
    2022-05-11 08:17:19,259 INFO: Train Loss:0.035 | Acc:0.9912 | F1:0.9723
    2022-05-11 08:17:28,947 INFO: val Loss:0.141 | Acc:0.9755 | F1:0.8795
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 08:19:18,923 INFO: Epoch:[166/200]
    2022-05-11 08:19:18,924 INFO: Train Loss:0.030 | Acc:0.9921 | F1:0.9772
    2022-05-11 08:19:28,606 INFO: val Loss:0.158 | Acc:0.9673 | F1:0.8635
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 08:21:18,458 INFO: Epoch:[167/200]
    2022-05-11 08:21:18,458 INFO: Train Loss:0.021 | Acc:0.9947 | F1:0.9810
    2022-05-11 08:21:28,116 INFO: val Loss:0.181 | Acc:0.9626 | F1:0.8506
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 08:23:17,948 INFO: Epoch:[168/200]
    2022-05-11 08:23:17,948 INFO: Train Loss:0.020 | Acc:0.9942 | F1:0.9826
    2022-05-11 08:23:27,589 INFO: val Loss:0.187 | Acc:0.9696 | F1:0.8741
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 08:25:17,485 INFO: Epoch:[169/200]
    2022-05-11 08:25:17,486 INFO: Train Loss:0.018 | Acc:0.9944 | F1:0.9828
    2022-05-11 08:25:27,108 INFO: val Loss:0.175 | Acc:0.9673 | F1:0.8697
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 08:27:17,039 INFO: Epoch:[170/200]
    2022-05-11 08:27:17,040 INFO: Train Loss:0.032 | Acc:0.9898 | F1:0.9706
    2022-05-11 08:27:26,740 INFO: val Loss:0.152 | Acc:0.9731 | F1:0.8927
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 08:29:16,591 INFO: Epoch:[171/200]
    2022-05-11 08:29:16,591 INFO: Train Loss:0.036 | Acc:0.9904 | F1:0.9668
    2022-05-11 08:29:26,278 INFO: val Loss:0.137 | Acc:0.9708 | F1:0.8712
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 08:31:15,961 INFO: Epoch:[172/200]
    2022-05-11 08:31:15,961 INFO: Train Loss:0.027 | Acc:0.9918 | F1:0.9696
    2022-05-11 08:31:25,660 INFO: val Loss:0.149 | Acc:0.9743 | F1:0.8930
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 08:33:15,429 INFO: Epoch:[173/200]
    2022-05-11 08:33:15,430 INFO: Train Loss:0.022 | Acc:0.9950 | F1:0.9845
    2022-05-11 08:33:25,139 INFO: val Loss:0.146 | Acc:0.9731 | F1:0.8724
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 08:35:15,046 INFO: Epoch:[174/200]
    2022-05-11 08:35:15,046 INFO: Train Loss:0.017 | Acc:0.9942 | F1:0.9815
    2022-05-11 08:35:24,738 INFO: val Loss:0.153 | Acc:0.9685 | F1:0.8587
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:50<00:00,  1.94it/s]
    2022-05-11 08:37:14,841 INFO: Epoch:[175/200]
    2022-05-11 08:37:14,842 INFO: Train Loss:0.025 | Acc:0.9921 | F1:0.9726
    2022-05-11 08:37:24,488 INFO: val Loss:0.143 | Acc:0.9755 | F1:0.8815
    2022-05-11 08:37:24,489 INFO: 
    Best Val Epoch:125 | Val Loss:0.1178 | Val Acc:0.9708 | Val F1:0.8478
    2022-05-11 08:37:24,489 INFO: Total Process time:354.451Minute
    2022-05-11 08:37:24,511 INFO: {'exp_num': '1', 'data_path': './data', 'Kfold': 5, 'model_path': 'results/', 'encoder_name': 'regnety_160', 'drop_path_rate': 0.2, 'img_size': 224, 'batch_size': 16, 'epochs': 200, 'optimizer': 'Lamb', 'initial_lr': 5e-06, 'weight_decay': 0.001, 'aug_ver': 2, 'scheduler': 'cycle', 'warm_epoch': 5, 'max_lr': 0.001, 'min_lr': 5e-06, 'tmax': 145, 'patience': 50, 'clipping': None, 'amp': True, 'multi_gpu': False, 'logging': False, 'num_workers': 0, 'seed': 42, 'fold': 1}
    

    <---- Training Params ---->
    Read train_df.csv
    Dataset size:3421
    Dataset size:856
    

    2022-05-11 08:37:25,318 INFO: Loading pretrained weights from url (https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth)
    2022-05-11 08:37:25,692 INFO: Computational complexity:       15.93 GMac
    2022-05-11 08:37:25,692 INFO: Number of parameters:           80.83 M 
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 08:39:15,174 INFO: Epoch:[001/200]
    2022-05-11 08:39:15,174 INFO: Train Loss:4.485 | Acc:0.0079 | F1:0.0037
    2022-05-11 08:39:24,897 INFO: val Loss:4.539 | Acc:0.0000 | F1:0.0000
    2022-05-11 08:39:26,560 INFO: -----------------SAVE:1epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.95it/s]
    2022-05-11 08:41:16,322 INFO: Epoch:[002/200]
    2022-05-11 08:41:16,322 INFO: Train Loss:4.471 | Acc:0.0111 | F1:0.0043
    2022-05-11 08:41:25,976 INFO: val Loss:4.514 | Acc:0.0000 | F1:0.0000
    2022-05-11 08:41:27,712 INFO: -----------------SAVE:2epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.96it/s]
    2022-05-11 08:43:17,088 INFO: Epoch:[003/200]
    2022-05-11 08:43:17,089 INFO: Train Loss:4.448 | Acc:0.0202 | F1:0.0056
    2022-05-11 08:43:26,787 INFO: val Loss:4.481 | Acc:0.0012 | F1:0.0002
    2022-05-11 08:43:28,499 INFO: -----------------SAVE:3epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.96it/s]
    2022-05-11 08:45:17,884 INFO: Epoch:[004/200]
    2022-05-11 08:45:17,884 INFO: Train Loss:4.406 | Acc:0.0327 | F1:0.0084
    2022-05-11 08:45:27,559 INFO: val Loss:4.443 | Acc:0.0047 | F1:0.0010
    2022-05-11 08:45:29,416 INFO: -----------------SAVE:4epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.96it/s]
    2022-05-11 08:47:18,779 INFO: Epoch:[005/200]
    2022-05-11 08:47:18,779 INFO: Train Loss:4.354 | Acc:0.0666 | F1:0.0143
    2022-05-11 08:47:28,460 INFO: val Loss:4.406 | Acc:0.0362 | F1:0.0049
    2022-05-11 08:47:30,302 INFO: -----------------SAVE:5epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.96it/s]
    2022-05-11 08:49:19,526 INFO: Epoch:[006/200]
    2022-05-11 08:49:19,526 INFO: Train Loss:4.022 | Acc:0.3642 | F1:0.0742
    2022-05-11 08:49:29,190 INFO: val Loss:3.809 | Acc:0.5771 | F1:0.1065
    2022-05-11 08:49:31,017 INFO: -----------------SAVE:6epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:49<00:00,  1.96it/s]
    2022-05-11 08:51:20,303 INFO: Epoch:[007/200]
    2022-05-11 08:51:20,303 INFO: Train Loss:3.250 | Acc:0.7469 | F1:0.1380
    2022-05-11 08:51:29,978 INFO: val Loss:2.836 | Acc:0.7652 | F1:0.1371
    2022-05-11 08:51:31,789 INFO: -----------------SAVE:7epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:52<00:00,  1.91it/s]
    2022-05-11 08:53:23,842 INFO: Epoch:[008/200]
    2022-05-11 08:53:23,842 INFO: Train Loss:2.395 | Acc:0.8232 | F1:0.1452
    2022-05-11 08:53:33,894 INFO: val Loss:1.762 | Acc:0.8364 | F1:0.1468
    2022-05-11 08:53:35,768 INFO: -----------------SAVE:8epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:51<00:00,  1.91it/s]
    2022-05-11 08:55:27,677 INFO: Epoch:[009/200]
    2022-05-11 08:55:27,677 INFO: Train Loss:1.621 | Acc:0.8325 | F1:0.1485
    2022-05-11 08:55:37,482 INFO: val Loss:1.076 | Acc:0.8411 | F1:0.1515
    2022-05-11 08:55:39,404 INFO: -----------------SAVE:9epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:52<00:00,  1.90it/s]
    2022-05-11 08:57:32,044 INFO: Epoch:[010/200]
    2022-05-11 08:57:32,044 INFO: Train Loss:1.248 | Acc:0.8416 | F1:0.1545
    2022-05-11 08:57:42,183 INFO: val Loss:1.009 | Acc:0.8493 | F1:0.1563
    2022-05-11 08:57:44,188 INFO: -----------------SAVE:10epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 08:59:40,029 INFO: Epoch:[011/200]
    2022-05-11 08:59:40,029 INFO: Train Loss:1.103 | Acc:0.8451 | F1:0.1556
    2022-05-11 08:59:50,190 INFO: val Loss:0.895 | Acc:0.8493 | F1:0.1563
    2022-05-11 08:59:52,009 INFO: -----------------SAVE:11epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 09:01:47,686 INFO: Epoch:[012/200]
    2022-05-11 09:01:47,686 INFO: Train Loss:0.978 | Acc:0.8471 | F1:0.1558
    2022-05-11 09:01:57,826 INFO: val Loss:0.801 | Acc:0.8493 | F1:0.1563
    2022-05-11 09:01:59,729 INFO: -----------------SAVE:12epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 09:03:56,021 INFO: Epoch:[013/200]
    2022-05-11 09:03:56,021 INFO: Train Loss:0.873 | Acc:0.8477 | F1:0.1559
    2022-05-11 09:04:05,947 INFO: val Loss:0.741 | Acc:0.8493 | F1:0.1562
    2022-05-11 09:04:07,901 INFO: -----------------SAVE:13epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 09:06:01,450 INFO: Epoch:[014/200]
    2022-05-11 09:06:01,450 INFO: Train Loss:0.808 | Acc:0.8468 | F1:0.1582
    2022-05-11 09:06:11,327 INFO: val Loss:0.675 | Acc:0.8528 | F1:0.1739
    2022-05-11 09:06:13,259 INFO: -----------------SAVE:14epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 09:08:07,029 INFO: Epoch:[015/200]
    2022-05-11 09:08:07,029 INFO: Train Loss:0.730 | Acc:0.8489 | F1:0.1658
    2022-05-11 09:08:16,916 INFO: val Loss:0.642 | Acc:0.8481 | F1:0.2021
    2022-05-11 09:08:18,823 INFO: -----------------SAVE:15epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 09:10:12,592 INFO: Epoch:[016/200]
    2022-05-11 09:10:12,592 INFO: Train Loss:0.682 | Acc:0.8515 | F1:0.1841
    2022-05-11 09:10:22,451 INFO: val Loss:0.631 | Acc:0.8551 | F1:0.2275
    2022-05-11 09:10:24,365 INFO: -----------------SAVE:16epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 09:12:17,783 INFO: Epoch:[017/200]
    2022-05-11 09:12:17,783 INFO: Train Loss:0.630 | Acc:0.8603 | F1:0.2335
    2022-05-11 09:12:27,655 INFO: val Loss:0.574 | Acc:0.8610 | F1:0.2562
    2022-05-11 09:12:29,596 INFO: -----------------SAVE:17epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 09:14:23,267 INFO: Epoch:[018/200]
    2022-05-11 09:14:23,267 INFO: Train Loss:0.578 | Acc:0.8711 | F1:0.2992
    2022-05-11 09:14:33,200 INFO: val Loss:0.492 | Acc:0.8843 | F1:0.3625
    2022-05-11 09:14:35,182 INFO: -----------------SAVE:18epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:54<00:00,  1.87it/s]
    2022-05-11 09:16:29,430 INFO: Epoch:[019/200]
    2022-05-11 09:16:29,430 INFO: Train Loss:0.530 | Acc:0.8784 | F1:0.3483
    2022-05-11 09:16:39,413 INFO: val Loss:0.463 | Acc:0.8902 | F1:0.3728
    2022-05-11 09:16:41,343 INFO: -----------------SAVE:19epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 09:18:35,210 INFO: Epoch:[020/200]
    2022-05-11 09:18:35,211 INFO: Train Loss:0.505 | Acc:0.8822 | F1:0.3571
    2022-05-11 09:18:45,098 INFO: val Loss:0.403 | Acc:0.8995 | F1:0.4063
    2022-05-11 09:18:47,037 INFO: -----------------SAVE:20epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 09:20:40,662 INFO: Epoch:[021/200]
    2022-05-11 09:20:40,662 INFO: Train Loss:0.460 | Acc:0.8892 | F1:0.4091
    2022-05-11 09:20:50,529 INFO: val Loss:0.374 | Acc:0.8925 | F1:0.4529
    2022-05-11 09:20:52,536 INFO: -----------------SAVE:21epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 09:22:46,235 INFO: Epoch:[022/200]
    2022-05-11 09:22:46,236 INFO: Train Loss:0.431 | Acc:0.8968 | F1:0.4359
    2022-05-11 09:22:56,169 INFO: val Loss:0.353 | Acc:0.9077 | F1:0.5008
    2022-05-11 09:22:58,004 INFO: -----------------SAVE:22epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 09:24:51,913 INFO: Epoch:[023/200]
    2022-05-11 09:24:51,913 INFO: Train Loss:0.402 | Acc:0.9030 | F1:0.4870
    2022-05-11 09:25:01,832 INFO: val Loss:0.314 | Acc:0.9159 | F1:0.5334
    2022-05-11 09:25:03,755 INFO: -----------------SAVE:23epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 09:26:57,557 INFO: Epoch:[024/200]
    2022-05-11 09:26:57,558 INFO: Train Loss:0.390 | Acc:0.9038 | F1:0.5014
    2022-05-11 09:27:07,537 INFO: val Loss:0.308 | Acc:0.9124 | F1:0.5607
    2022-05-11 09:27:09,503 INFO: -----------------SAVE:24epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 09:29:03,281 INFO: Epoch:[025/200]
    2022-05-11 09:29:03,281 INFO: Train Loss:0.358 | Acc:0.9088 | F1:0.5336
    2022-05-11 09:29:13,167 INFO: val Loss:0.313 | Acc:0.9171 | F1:0.5461
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 09:31:06,667 INFO: Epoch:[026/200]
    2022-05-11 09:31:06,667 INFO: Train Loss:0.339 | Acc:0.9132 | F1:0.5682
    2022-05-11 09:31:16,525 INFO: val Loss:0.276 | Acc:0.9264 | F1:0.5839
    2022-05-11 09:31:18,459 INFO: -----------------SAVE:26epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 09:33:11,952 INFO: Epoch:[027/200]
    2022-05-11 09:33:11,952 INFO: Train Loss:0.327 | Acc:0.9158 | F1:0.5763
    2022-05-11 09:33:21,843 INFO: val Loss:0.290 | Acc:0.9229 | F1:0.5802
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 09:35:15,541 INFO: Epoch:[028/200]
    2022-05-11 09:35:15,541 INFO: Train Loss:0.303 | Acc:0.9199 | F1:0.6000
    2022-05-11 09:35:25,377 INFO: val Loss:0.253 | Acc:0.9276 | F1:0.5977
    2022-05-11 09:35:27,238 INFO: -----------------SAVE:28epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 09:37:21,145 INFO: Epoch:[029/200]
    2022-05-11 09:37:21,145 INFO: Train Loss:0.294 | Acc:0.9225 | F1:0.6193
    2022-05-11 09:37:31,010 INFO: val Loss:0.239 | Acc:0.9287 | F1:0.5989
    2022-05-11 09:37:32,831 INFO: -----------------SAVE:29epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 09:39:26,464 INFO: Epoch:[030/200]
    2022-05-11 09:39:26,464 INFO: Train Loss:0.287 | Acc:0.9272 | F1:0.6572
    2022-05-11 09:39:36,342 INFO: val Loss:0.226 | Acc:0.9334 | F1:0.6260
    2022-05-11 09:39:38,204 INFO: -----------------SAVE:30epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 09:41:31,932 INFO: Epoch:[031/200]
    2022-05-11 09:41:31,932 INFO: Train Loss:0.281 | Acc:0.9258 | F1:0.6536
    2022-05-11 09:41:41,856 INFO: val Loss:0.270 | Acc:0.9346 | F1:0.6340
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 09:43:35,287 INFO: Epoch:[032/200]
    2022-05-11 09:43:35,287 INFO: Train Loss:0.270 | Acc:0.9296 | F1:0.6620
    2022-05-11 09:43:45,180 INFO: val Loss:0.234 | Acc:0.9346 | F1:0.6354
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 09:45:38,394 INFO: Epoch:[033/200]
    2022-05-11 09:45:38,394 INFO: Train Loss:0.247 | Acc:0.9322 | F1:0.6855
    2022-05-11 09:45:48,267 INFO: val Loss:0.290 | Acc:0.9089 | F1:0.6487
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 09:47:41,878 INFO: Epoch:[034/200]
    2022-05-11 09:47:41,878 INFO: Train Loss:0.248 | Acc:0.9304 | F1:0.6987
    2022-05-11 09:47:51,781 INFO: val Loss:0.208 | Acc:0.9463 | F1:0.6802
    2022-05-11 09:47:53,771 INFO: -----------------SAVE:34epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 09:49:47,347 INFO: Epoch:[035/200]
    2022-05-11 09:49:47,348 INFO: Train Loss:0.244 | Acc:0.9372 | F1:0.7248
    2022-05-11 09:49:57,231 INFO: val Loss:0.205 | Acc:0.9369 | F1:0.6776
    2022-05-11 09:49:59,072 INFO: -----------------SAVE:35epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 09:51:52,530 INFO: Epoch:[036/200]
    2022-05-11 09:51:52,531 INFO: Train Loss:0.234 | Acc:0.9392 | F1:0.7424
    2022-05-11 09:52:02,447 INFO: val Loss:0.242 | Acc:0.9322 | F1:0.6796
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 09:53:55,841 INFO: Epoch:[037/200]
    2022-05-11 09:53:55,842 INFO: Train Loss:0.223 | Acc:0.9401 | F1:0.7324
    2022-05-11 09:54:05,767 INFO: val Loss:0.218 | Acc:0.9381 | F1:0.6431
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 09:55:59,348 INFO: Epoch:[038/200]
    2022-05-11 09:55:59,349 INFO: Train Loss:0.220 | Acc:0.9415 | F1:0.7422
    2022-05-11 09:56:09,281 INFO: val Loss:0.229 | Acc:0.9404 | F1:0.6725
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 09:58:02,688 INFO: Epoch:[039/200]
    2022-05-11 09:58:02,688 INFO: Train Loss:0.220 | Acc:0.9412 | F1:0.7445
    2022-05-11 09:58:12,542 INFO: val Loss:0.248 | Acc:0.9416 | F1:0.7035
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 10:00:05,937 INFO: Epoch:[040/200]
    2022-05-11 10:00:05,938 INFO: Train Loss:0.219 | Acc:0.9450 | F1:0.7623
    2022-05-11 10:00:15,814 INFO: val Loss:0.209 | Acc:0.9498 | F1:0.6947
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 10:02:09,372 INFO: Epoch:[041/200]
    2022-05-11 10:02:09,372 INFO: Train Loss:0.196 | Acc:0.9465 | F1:0.7716
    2022-05-11 10:02:19,277 INFO: val Loss:0.203 | Acc:0.9474 | F1:0.7215
    2022-05-11 10:02:21,178 INFO: -----------------SAVE:41epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 10:04:14,844 INFO: Epoch:[042/200]
    2022-05-11 10:04:14,844 INFO: Train Loss:0.196 | Acc:0.9483 | F1:0.7768
    2022-05-11 10:04:24,756 INFO: val Loss:0.230 | Acc:0.9322 | F1:0.7026
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 10:06:18,370 INFO: Epoch:[043/200]
    2022-05-11 10:06:18,371 INFO: Train Loss:0.204 | Acc:0.9442 | F1:0.7562
    2022-05-11 10:06:28,278 INFO: val Loss:0.202 | Acc:0.9439 | F1:0.6773
    2022-05-11 10:06:30,268 INFO: -----------------SAVE:43epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 10:08:23,886 INFO: Epoch:[044/200]
    2022-05-11 10:08:23,886 INFO: Train Loss:0.194 | Acc:0.9468 | F1:0.7865
    2022-05-11 10:08:33,765 INFO: val Loss:0.192 | Acc:0.9451 | F1:0.6987
    2022-05-11 10:08:35,900 INFO: -----------------SAVE:44epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 10:10:29,770 INFO: Epoch:[045/200]
    2022-05-11 10:10:29,771 INFO: Train Loss:0.198 | Acc:0.9509 | F1:0.7966
    2022-05-11 10:10:39,685 INFO: val Loss:0.217 | Acc:0.9521 | F1:0.7446
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 10:12:33,324 INFO: Epoch:[046/200]
    2022-05-11 10:12:33,325 INFO: Train Loss:0.209 | Acc:0.9483 | F1:0.7859
    2022-05-11 10:12:43,162 INFO: val Loss:0.157 | Acc:0.9579 | F1:0.7301
    2022-05-11 10:12:45,034 INFO: -----------------SAVE:46epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 10:14:38,518 INFO: Epoch:[047/200]
    2022-05-11 10:14:38,518 INFO: Train Loss:0.185 | Acc:0.9497 | F1:0.7869
    2022-05-11 10:14:48,379 INFO: val Loss:0.190 | Acc:0.9533 | F1:0.7406
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 10:16:42,148 INFO: Epoch:[048/200]
    2022-05-11 10:16:42,148 INFO: Train Loss:0.190 | Acc:0.9497 | F1:0.8063
    2022-05-11 10:16:52,055 INFO: val Loss:0.235 | Acc:0.9463 | F1:0.7611
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 10:18:45,501 INFO: Epoch:[049/200]
    2022-05-11 10:18:45,502 INFO: Train Loss:0.193 | Acc:0.9509 | F1:0.8037
    2022-05-11 10:18:55,384 INFO: val Loss:0.173 | Acc:0.9568 | F1:0.7477
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 10:20:48,996 INFO: Epoch:[050/200]
    2022-05-11 10:20:48,996 INFO: Train Loss:0.186 | Acc:0.9538 | F1:0.8138
    2022-05-11 10:20:58,889 INFO: val Loss:0.211 | Acc:0.9579 | F1:0.7559
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 10:22:52,550 INFO: Epoch:[051/200]
    2022-05-11 10:22:52,551 INFO: Train Loss:0.179 | Acc:0.9494 | F1:0.7904
    2022-05-11 10:23:02,486 INFO: val Loss:0.171 | Acc:0.9579 | F1:0.7586
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 10:24:56,246 INFO: Epoch:[052/200]
    2022-05-11 10:24:56,246 INFO: Train Loss:0.197 | Acc:0.9518 | F1:0.8038
    2022-05-11 10:25:06,133 INFO: val Loss:0.195 | Acc:0.9614 | F1:0.7629
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 10:26:59,377 INFO: Epoch:[053/200]
    2022-05-11 10:26:59,377 INFO: Train Loss:0.166 | Acc:0.9529 | F1:0.8070
    2022-05-11 10:27:09,295 INFO: val Loss:0.179 | Acc:0.9579 | F1:0.7353
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 10:29:02,918 INFO: Epoch:[054/200]
    2022-05-11 10:29:02,919 INFO: Train Loss:0.190 | Acc:0.9521 | F1:0.8029
    2022-05-11 10:29:12,800 INFO: val Loss:0.170 | Acc:0.9603 | F1:0.7577
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:54<00:00,  1.87it/s]
    2022-05-11 10:31:07,333 INFO: Epoch:[055/200]
    2022-05-11 10:31:07,333 INFO: Train Loss:0.179 | Acc:0.9532 | F1:0.8099
    2022-05-11 10:31:17,237 INFO: val Loss:0.203 | Acc:0.9568 | F1:0.7504
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 10:33:10,705 INFO: Epoch:[056/200]
    2022-05-11 10:33:10,706 INFO: Train Loss:0.175 | Acc:0.9544 | F1:0.8152
    2022-05-11 10:33:20,555 INFO: val Loss:0.171 | Acc:0.9603 | F1:0.7918
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 10:35:14,319 INFO: Epoch:[057/200]
    2022-05-11 10:35:14,320 INFO: Train Loss:0.170 | Acc:0.9564 | F1:0.8300
    2022-05-11 10:35:24,178 INFO: val Loss:0.225 | Acc:0.9311 | F1:0.7700
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 10:37:17,636 INFO: Epoch:[058/200]
    2022-05-11 10:37:17,637 INFO: Train Loss:0.162 | Acc:0.9564 | F1:0.8289
    2022-05-11 10:37:27,490 INFO: val Loss:0.379 | Acc:0.9030 | F1:0.7599
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 10:39:21,037 INFO: Epoch:[059/200]
    2022-05-11 10:39:21,037 INFO: Train Loss:0.167 | Acc:0.9582 | F1:0.8320
    2022-05-11 10:39:30,943 INFO: val Loss:0.205 | Acc:0.9451 | F1:0.7214
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 10:41:24,503 INFO: Epoch:[060/200]
    2022-05-11 10:41:24,503 INFO: Train Loss:0.154 | Acc:0.9602 | F1:0.8374
    2022-05-11 10:41:34,372 INFO: val Loss:0.258 | Acc:0.9393 | F1:0.7290
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 10:43:27,893 INFO: Epoch:[061/200]
    2022-05-11 10:43:27,893 INFO: Train Loss:0.194 | Acc:0.9512 | F1:0.8146
    2022-05-11 10:43:37,737 INFO: val Loss:0.193 | Acc:0.9439 | F1:0.7043
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 10:45:31,231 INFO: Epoch:[062/200]
    2022-05-11 10:45:31,232 INFO: Train Loss:0.159 | Acc:0.9579 | F1:0.8237
    2022-05-11 10:45:41,122 INFO: val Loss:0.212 | Acc:0.9533 | F1:0.7337
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 10:47:34,544 INFO: Epoch:[063/200]
    2022-05-11 10:47:34,544 INFO: Train Loss:0.164 | Acc:0.9562 | F1:0.8305
    2022-05-11 10:47:44,412 INFO: val Loss:0.199 | Acc:0.9509 | F1:0.7252
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 10:49:38,132 INFO: Epoch:[064/200]
    2022-05-11 10:49:38,132 INFO: Train Loss:0.145 | Acc:0.9620 | F1:0.8455
    2022-05-11 10:49:48,002 INFO: val Loss:0.211 | Acc:0.9439 | F1:0.6852
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 10:51:41,701 INFO: Epoch:[065/200]
    2022-05-11 10:51:41,701 INFO: Train Loss:0.160 | Acc:0.9576 | F1:0.8456
    2022-05-11 10:51:51,695 INFO: val Loss:0.149 | Acc:0.9579 | F1:0.7768
    2022-05-11 10:51:53,662 INFO: -----------------SAVE:65epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 10:53:47,521 INFO: Epoch:[066/200]
    2022-05-11 10:53:47,521 INFO: Train Loss:0.165 | Acc:0.9579 | F1:0.8401
    2022-05-11 10:53:57,399 INFO: val Loss:0.209 | Acc:0.9451 | F1:0.7327
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 10:55:50,912 INFO: Epoch:[067/200]
    2022-05-11 10:55:50,913 INFO: Train Loss:0.154 | Acc:0.9579 | F1:0.8415
    2022-05-11 10:56:00,819 INFO: val Loss:0.256 | Acc:0.9498 | F1:0.7495
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 10:57:54,453 INFO: Epoch:[068/200]
    2022-05-11 10:57:54,454 INFO: Train Loss:0.159 | Acc:0.9597 | F1:0.8473
    2022-05-11 10:58:04,329 INFO: val Loss:0.202 | Acc:0.9603 | F1:0.8050
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 10:59:57,745 INFO: Epoch:[069/200]
    2022-05-11 10:59:57,745 INFO: Train Loss:0.145 | Acc:0.9597 | F1:0.8417
    2022-05-11 11:00:07,617 INFO: val Loss:0.171 | Acc:0.9591 | F1:0.7768
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 11:02:01,022 INFO: Epoch:[070/200]
    2022-05-11 11:02:01,023 INFO: Train Loss:0.134 | Acc:0.9649 | F1:0.8580
    2022-05-11 11:02:10,896 INFO: val Loss:0.223 | Acc:0.9509 | F1:0.7349
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 11:04:04,338 INFO: Epoch:[071/200]
    2022-05-11 11:04:04,339 INFO: Train Loss:0.138 | Acc:0.9608 | F1:0.8422
    2022-05-11 11:04:14,197 INFO: val Loss:0.180 | Acc:0.9591 | F1:0.7821
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 11:06:07,466 INFO: Epoch:[072/200]
    2022-05-11 11:06:07,467 INFO: Train Loss:0.140 | Acc:0.9617 | F1:0.8521
    2022-05-11 11:06:17,316 INFO: val Loss:0.170 | Acc:0.9544 | F1:0.7497
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 11:08:10,771 INFO: Epoch:[073/200]
    2022-05-11 11:08:10,772 INFO: Train Loss:0.142 | Acc:0.9661 | F1:0.8780
    2022-05-11 11:08:20,614 INFO: val Loss:0.146 | Acc:0.9591 | F1:0.7955
    2022-05-11 11:08:22,474 INFO: -----------------SAVE:73epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 11:10:16,046 INFO: Epoch:[074/200]
    2022-05-11 11:10:16,046 INFO: Train Loss:0.135 | Acc:0.9635 | F1:0.8581
    2022-05-11 11:10:25,899 INFO: val Loss:0.195 | Acc:0.9556 | F1:0.7767
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 11:12:19,071 INFO: Epoch:[075/200]
    2022-05-11 11:12:19,071 INFO: Train Loss:0.136 | Acc:0.9597 | F1:0.8395
    2022-05-11 11:12:28,902 INFO: val Loss:0.206 | Acc:0.9463 | F1:0.7290
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 11:14:22,186 INFO: Epoch:[076/200]
    2022-05-11 11:14:22,186 INFO: Train Loss:0.127 | Acc:0.9652 | F1:0.8688
    2022-05-11 11:14:32,043 INFO: val Loss:0.210 | Acc:0.9544 | F1:0.7510
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 11:16:25,545 INFO: Epoch:[077/200]
    2022-05-11 11:16:25,546 INFO: Train Loss:0.135 | Acc:0.9640 | F1:0.8688
    2022-05-11 11:16:35,398 INFO: val Loss:0.206 | Acc:0.9521 | F1:0.7095
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 11:18:28,687 INFO: Epoch:[078/200]
    2022-05-11 11:18:28,688 INFO: Train Loss:0.125 | Acc:0.9652 | F1:0.8637
    2022-05-11 11:18:38,559 INFO: val Loss:0.207 | Acc:0.9509 | F1:0.7448
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 11:20:32,070 INFO: Epoch:[079/200]
    2022-05-11 11:20:32,071 INFO: Train Loss:0.132 | Acc:0.9649 | F1:0.8648
    2022-05-11 11:20:41,928 INFO: val Loss:0.284 | Acc:0.9381 | F1:0.6624
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 11:22:35,381 INFO: Epoch:[080/200]
    2022-05-11 11:22:35,381 INFO: Train Loss:0.132 | Acc:0.9643 | F1:0.8725
    2022-05-11 11:22:45,205 INFO: val Loss:0.205 | Acc:0.9556 | F1:0.7442
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 11:24:39,114 INFO: Epoch:[081/200]
    2022-05-11 11:24:39,114 INFO: Train Loss:0.107 | Acc:0.9676 | F1:0.8690
    2022-05-11 11:24:48,987 INFO: val Loss:0.198 | Acc:0.9568 | F1:0.7661
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 11:26:42,437 INFO: Epoch:[082/200]
    2022-05-11 11:26:42,437 INFO: Train Loss:0.119 | Acc:0.9693 | F1:0.8917
    2022-05-11 11:26:52,267 INFO: val Loss:0.176 | Acc:0.9661 | F1:0.8072
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 11:28:45,936 INFO: Epoch:[083/200]
    2022-05-11 11:28:45,936 INFO: Train Loss:0.124 | Acc:0.9667 | F1:0.8665
    2022-05-11 11:28:55,789 INFO: val Loss:0.177 | Acc:0.9603 | F1:0.7903
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 11:30:49,303 INFO: Epoch:[084/200]
    2022-05-11 11:30:49,304 INFO: Train Loss:0.114 | Acc:0.9676 | F1:0.8916
    2022-05-11 11:30:59,131 INFO: val Loss:0.204 | Acc:0.9533 | F1:0.7679
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 11:32:52,609 INFO: Epoch:[085/200]
    2022-05-11 11:32:52,610 INFO: Train Loss:0.108 | Acc:0.9711 | F1:0.8879
    2022-05-11 11:33:02,492 INFO: val Loss:0.213 | Acc:0.9544 | F1:0.7620
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 11:34:55,889 INFO: Epoch:[086/200]
    2022-05-11 11:34:55,889 INFO: Train Loss:0.125 | Acc:0.9655 | F1:0.8700
    2022-05-11 11:35:05,786 INFO: val Loss:0.207 | Acc:0.9521 | F1:0.7785
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 11:36:59,277 INFO: Epoch:[087/200]
    2022-05-11 11:36:59,277 INFO: Train Loss:0.106 | Acc:0.9690 | F1:0.8850
    2022-05-11 11:37:09,201 INFO: val Loss:0.213 | Acc:0.9568 | F1:0.7499
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 11:39:02,360 INFO: Epoch:[088/200]
    2022-05-11 11:39:02,360 INFO: Train Loss:0.111 | Acc:0.9711 | F1:0.8949
    2022-05-11 11:39:12,212 INFO: val Loss:0.204 | Acc:0.9614 | F1:0.7927
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 11:41:05,739 INFO: Epoch:[089/200]
    2022-05-11 11:41:05,739 INFO: Train Loss:0.109 | Acc:0.9678 | F1:0.8745
    2022-05-11 11:41:15,621 INFO: val Loss:0.172 | Acc:0.9603 | F1:0.7949
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 11:43:08,879 INFO: Epoch:[090/200]
    2022-05-11 11:43:08,879 INFO: Train Loss:0.089 | Acc:0.9752 | F1:0.9002
    2022-05-11 11:43:18,735 INFO: val Loss:0.203 | Acc:0.9626 | F1:0.8099
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 11:45:12,323 INFO: Epoch:[091/200]
    2022-05-11 11:45:12,323 INFO: Train Loss:0.113 | Acc:0.9684 | F1:0.8820
    2022-05-11 11:45:22,174 INFO: val Loss:0.193 | Acc:0.9591 | F1:0.7820
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 11:47:15,747 INFO: Epoch:[092/200]
    2022-05-11 11:47:15,747 INFO: Train Loss:0.101 | Acc:0.9722 | F1:0.8975
    2022-05-11 11:47:25,600 INFO: val Loss:0.245 | Acc:0.9439 | F1:0.7223
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 11:49:19,141 INFO: Epoch:[093/200]
    2022-05-11 11:49:19,141 INFO: Train Loss:0.105 | Acc:0.9716 | F1:0.8909
    2022-05-11 11:49:29,024 INFO: val Loss:0.814 | Acc:0.8995 | F1:0.7306
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 11:51:22,713 INFO: Epoch:[094/200]
    2022-05-11 11:51:22,714 INFO: Train Loss:0.114 | Acc:0.9705 | F1:0.8902
    2022-05-11 11:51:32,685 INFO: val Loss:0.164 | Acc:0.9650 | F1:0.8216
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 11:53:26,244 INFO: Epoch:[095/200]
    2022-05-11 11:53:26,244 INFO: Train Loss:0.093 | Acc:0.9752 | F1:0.9116
    2022-05-11 11:53:36,105 INFO: val Loss:0.205 | Acc:0.9603 | F1:0.7943
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 11:55:29,460 INFO: Epoch:[096/200]
    2022-05-11 11:55:29,461 INFO: Train Loss:0.096 | Acc:0.9757 | F1:0.9082
    2022-05-11 11:55:39,313 INFO: val Loss:0.239 | Acc:0.9428 | F1:0.7512
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 11:57:32,416 INFO: Epoch:[097/200]
    2022-05-11 11:57:32,416 INFO: Train Loss:0.094 | Acc:0.9746 | F1:0.9127
    2022-05-11 11:57:42,278 INFO: val Loss:0.241 | Acc:0.9521 | F1:0.7298
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 11:59:35,648 INFO: Epoch:[098/200]
    2022-05-11 11:59:35,648 INFO: Train Loss:0.097 | Acc:0.9740 | F1:0.9034
    2022-05-11 11:59:45,506 INFO: val Loss:0.267 | Acc:0.9393 | F1:0.6961
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 12:01:38,844 INFO: Epoch:[099/200]
    2022-05-11 12:01:38,845 INFO: Train Loss:0.097 | Acc:0.9731 | F1:0.8885
    2022-05-11 12:01:48,714 INFO: val Loss:0.171 | Acc:0.9614 | F1:0.8108
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 12:03:42,190 INFO: Epoch:[100/200]
    2022-05-11 12:03:42,190 INFO: Train Loss:0.089 | Acc:0.9760 | F1:0.9230
    2022-05-11 12:03:52,074 INFO: val Loss:0.437 | Acc:0.9065 | F1:0.7194
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 12:05:45,468 INFO: Epoch:[101/200]
    2022-05-11 12:05:45,468 INFO: Train Loss:0.086 | Acc:0.9766 | F1:0.9178
    2022-05-11 12:05:55,317 INFO: val Loss:0.194 | Acc:0.9568 | F1:0.7953
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 12:07:48,682 INFO: Epoch:[102/200]
    2022-05-11 12:07:48,683 INFO: Train Loss:0.094 | Acc:0.9746 | F1:0.9115
    2022-05-11 12:07:58,562 INFO: val Loss:0.202 | Acc:0.9568 | F1:0.7801
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 12:09:51,938 INFO: Epoch:[103/200]
    2022-05-11 12:09:51,938 INFO: Train Loss:0.090 | Acc:0.9749 | F1:0.9159
    2022-05-11 12:10:01,837 INFO: val Loss:0.194 | Acc:0.9650 | F1:0.7954
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 12:11:55,424 INFO: Epoch:[104/200]
    2022-05-11 12:11:55,424 INFO: Train Loss:0.095 | Acc:0.9731 | F1:0.9010
    2022-05-11 12:12:05,332 INFO: val Loss:0.232 | Acc:0.9474 | F1:0.7440
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 12:13:58,668 INFO: Epoch:[105/200]
    2022-05-11 12:13:58,668 INFO: Train Loss:0.079 | Acc:0.9766 | F1:0.9159
    2022-05-11 12:14:08,579 INFO: val Loss:0.194 | Acc:0.9591 | F1:0.7753
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 12:16:02,341 INFO: Epoch:[106/200]
    2022-05-11 12:16:02,341 INFO: Train Loss:0.084 | Acc:0.9760 | F1:0.9174
    2022-05-11 12:16:12,210 INFO: val Loss:0.205 | Acc:0.9486 | F1:0.7548
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 12:18:05,906 INFO: Epoch:[107/200]
    2022-05-11 12:18:05,907 INFO: Train Loss:0.071 | Acc:0.9781 | F1:0.9192
    2022-05-11 12:18:15,737 INFO: val Loss:0.210 | Acc:0.9568 | F1:0.7668
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 12:20:09,065 INFO: Epoch:[108/200]
    2022-05-11 12:20:09,065 INFO: Train Loss:0.100 | Acc:0.9737 | F1:0.9113
    2022-05-11 12:20:18,954 INFO: val Loss:0.180 | Acc:0.9626 | F1:0.8001
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 12:22:12,444 INFO: Epoch:[109/200]
    2022-05-11 12:22:12,445 INFO: Train Loss:0.090 | Acc:0.9766 | F1:0.9209
    2022-05-11 12:22:22,281 INFO: val Loss:0.205 | Acc:0.9556 | F1:0.7783
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 12:24:15,737 INFO: Epoch:[110/200]
    2022-05-11 12:24:15,737 INFO: Train Loss:0.086 | Acc:0.9757 | F1:0.9188
    2022-05-11 12:24:25,611 INFO: val Loss:0.193 | Acc:0.9579 | F1:0.7714
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 12:26:19,253 INFO: Epoch:[111/200]
    2022-05-11 12:26:19,253 INFO: Train Loss:0.070 | Acc:0.9807 | F1:0.9345
    2022-05-11 12:26:29,101 INFO: val Loss:0.206 | Acc:0.9568 | F1:0.7855
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 12:28:23,057 INFO: Epoch:[112/200]
    2022-05-11 12:28:23,058 INFO: Train Loss:0.091 | Acc:0.9719 | F1:0.8953
    2022-05-11 12:28:32,925 INFO: val Loss:0.230 | Acc:0.9614 | F1:0.7985
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 12:30:26,291 INFO: Epoch:[113/200]
    2022-05-11 12:30:26,292 INFO: Train Loss:0.077 | Acc:0.9763 | F1:0.9192
    2022-05-11 12:30:36,176 INFO: val Loss:0.250 | Acc:0.9579 | F1:0.7719
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 12:32:29,593 INFO: Epoch:[114/200]
    2022-05-11 12:32:29,594 INFO: Train Loss:0.085 | Acc:0.9778 | F1:0.9085
    2022-05-11 12:32:39,471 INFO: val Loss:0.225 | Acc:0.9568 | F1:0.7441
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 12:34:32,967 INFO: Epoch:[115/200]
    2022-05-11 12:34:32,968 INFO: Train Loss:0.084 | Acc:0.9798 | F1:0.9301
    2022-05-11 12:34:42,890 INFO: val Loss:0.220 | Acc:0.9521 | F1:0.7476
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 12:36:36,718 INFO: Epoch:[116/200]
    2022-05-11 12:36:36,719 INFO: Train Loss:0.071 | Acc:0.9810 | F1:0.9333
    2022-05-11 12:36:46,603 INFO: val Loss:0.253 | Acc:0.9521 | F1:0.7779
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 12:38:40,123 INFO: Epoch:[117/200]
    2022-05-11 12:38:40,123 INFO: Train Loss:0.075 | Acc:0.9813 | F1:0.9330
    2022-05-11 12:38:50,021 INFO: val Loss:0.250 | Acc:0.9509 | F1:0.7721
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 12:40:43,470 INFO: Epoch:[118/200]
    2022-05-11 12:40:43,471 INFO: Train Loss:0.085 | Acc:0.9772 | F1:0.9179
    2022-05-11 12:40:53,385 INFO: val Loss:0.219 | Acc:0.9591 | F1:0.7916
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 12:42:47,005 INFO: Epoch:[119/200]
    2022-05-11 12:42:47,006 INFO: Train Loss:0.071 | Acc:0.9787 | F1:0.9275
    2022-05-11 12:42:56,883 INFO: val Loss:0.162 | Acc:0.9661 | F1:0.8229
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 12:44:50,547 INFO: Epoch:[120/200]
    2022-05-11 12:44:50,547 INFO: Train Loss:0.066 | Acc:0.9819 | F1:0.9375
    2022-05-11 12:45:00,413 INFO: val Loss:0.212 | Acc:0.9638 | F1:0.7895
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.88it/s]
    2022-05-11 12:46:54,274 INFO: Epoch:[121/200]
    2022-05-11 12:46:54,274 INFO: Train Loss:0.071 | Acc:0.9810 | F1:0.9346
    2022-05-11 12:47:04,193 INFO: val Loss:0.211 | Acc:0.9626 | F1:0.7936
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 12:48:57,550 INFO: Epoch:[122/200]
    2022-05-11 12:48:57,550 INFO: Train Loss:0.072 | Acc:0.9801 | F1:0.9306
    2022-05-11 12:49:07,446 INFO: val Loss:0.225 | Acc:0.9544 | F1:0.7559
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 12:51:00,750 INFO: Epoch:[123/200]
    2022-05-11 12:51:00,751 INFO: Train Loss:0.065 | Acc:0.9833 | F1:0.9465
    2022-05-11 12:51:10,645 INFO: val Loss:0.236 | Acc:0.9568 | F1:0.7702
    2022-05-11 12:51:10,646 INFO: 
    Best Val Epoch:73 | Val Loss:0.1460 | Val Acc:0.9591 | Val F1:0.7955
    2022-05-11 12:51:10,646 INFO: Total Process time:253.749Minute
    2022-05-11 12:51:10,674 INFO: {'exp_num': '2', 'data_path': './data', 'Kfold': 5, 'model_path': 'results/', 'encoder_name': 'regnety_160', 'drop_path_rate': 0.2, 'img_size': 224, 'batch_size': 16, 'epochs': 200, 'optimizer': 'Lamb', 'initial_lr': 5e-06, 'weight_decay': 0.001, 'aug_ver': 2, 'scheduler': 'cycle', 'warm_epoch': 5, 'max_lr': 0.001, 'min_lr': 5e-06, 'tmax': 145, 'patience': 50, 'clipping': None, 'amp': True, 'multi_gpu': False, 'logging': False, 'num_workers': 0, 'seed': 42, 'fold': 2}
    

    <---- Training Params ---->
    Read train_df.csv
    Dataset size:3422
    Dataset size:855
    

    2022-05-11 12:51:11,501 INFO: Loading pretrained weights from url (https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth)
    2022-05-11 12:51:11,876 INFO: Computational complexity:       15.93 GMac
    2022-05-11 12:51:11,877 INFO: Number of parameters:           80.83 M 
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-11 12:53:08,846 INFO: Epoch:[001/200]
    2022-05-11 12:53:08,846 INFO: Train Loss:4.487 | Acc:0.0091 | F1:0.0048
    2022-05-11 12:53:19,908 INFO: val Loss:4.533 | Acc:0.0000 | F1:0.0000
    2022-05-11 12:53:21,639 INFO: -----------------SAVE:1epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:53<00:00,  1.89it/s]
    2022-05-11 12:55:15,048 INFO: Epoch:[002/200]
    2022-05-11 12:55:15,049 INFO: Train Loss:4.469 | Acc:0.0132 | F1:0.0044
    2022-05-11 12:55:25,021 INFO: val Loss:4.519 | Acc:0.0012 | F1:0.0002
    2022-05-11 12:55:26,843 INFO: -----------------SAVE:2epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:52<00:00,  1.90it/s]
    2022-05-11 12:57:19,710 INFO: Epoch:[003/200]
    2022-05-11 12:57:19,710 INFO: Train Loss:4.446 | Acc:0.0132 | F1:0.0036
    2022-05-11 12:57:30,419 INFO: val Loss:4.489 | Acc:0.0047 | F1:0.0007
    2022-05-11 12:57:32,241 INFO: -----------------SAVE:3epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 12:59:30,443 INFO: Epoch:[004/200]
    2022-05-11 12:59:30,443 INFO: Train Loss:4.398 | Acc:0.0359 | F1:0.0093
    2022-05-11 12:59:41,046 INFO: val Loss:4.441 | Acc:0.0070 | F1:0.0013
    2022-05-11 12:59:42,740 INFO: -----------------SAVE:4epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 13:01:40,837 INFO: Epoch:[005/200]
    2022-05-11 13:01:40,838 INFO: Train Loss:4.348 | Acc:0.0687 | F1:0.0144
    2022-05-11 13:01:51,398 INFO: val Loss:4.383 | Acc:0.0339 | F1:0.0060
    2022-05-11 13:01:53,245 INFO: -----------------SAVE:5epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 13:03:51,370 INFO: Epoch:[006/200]
    2022-05-11 13:03:51,370 INFO: Train Loss:4.013 | Acc:0.3828 | F1:0.0770
    2022-05-11 13:04:02,018 INFO: val Loss:3.705 | Acc:0.6094 | F1:0.1112
    2022-05-11 13:04:03,760 INFO: -----------------SAVE:6epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 13:06:01,925 INFO: Epoch:[007/200]
    2022-05-11 13:06:01,925 INFO: Train Loss:3.226 | Acc:0.7557 | F1:0.1375
    2022-05-11 13:06:12,462 INFO: val Loss:2.768 | Acc:0.7708 | F1:0.1352
    2022-05-11 13:06:14,309 INFO: -----------------SAVE:7epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 13:08:12,582 INFO: Epoch:[008/200]
    2022-05-11 13:08:12,582 INFO: Train Loss:2.355 | Acc:0.8235 | F1:0.1452
    2022-05-11 13:08:22,997 INFO: val Loss:1.717 | Acc:0.8351 | F1:0.1449
    2022-05-11 13:08:24,699 INFO: -----------------SAVE:8epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 13:10:22,903 INFO: Epoch:[009/200]
    2022-05-11 13:10:22,903 INFO: Train Loss:1.588 | Acc:0.8331 | F1:0.1478
    2022-05-11 13:10:33,466 INFO: val Loss:1.087 | Acc:0.8491 | F1:0.1562
    2022-05-11 13:10:35,414 INFO: -----------------SAVE:9epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 13:12:33,540 INFO: Epoch:[010/200]
    2022-05-11 13:12:33,541 INFO: Train Loss:1.245 | Acc:0.8407 | F1:0.1532
    2022-05-11 13:12:44,138 INFO: val Loss:1.008 | Acc:0.8491 | F1:0.1562
    2022-05-11 13:12:45,871 INFO: -----------------SAVE:10epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 13:14:44,323 INFO: Epoch:[011/200]
    2022-05-11 13:14:44,323 INFO: Train Loss:1.094 | Acc:0.8445 | F1:0.1552
    2022-05-11 13:14:54,893 INFO: val Loss:0.895 | Acc:0.8491 | F1:0.1561
    2022-05-11 13:14:56,848 INFO: -----------------SAVE:11epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 13:16:54,996 INFO: Epoch:[012/200]
    2022-05-11 13:16:54,997 INFO: Train Loss:0.967 | Acc:0.8475 | F1:0.1560
    2022-05-11 13:17:05,627 INFO: val Loss:0.849 | Acc:0.8491 | F1:0.1561
    2022-05-11 13:17:07,658 INFO: -----------------SAVE:12epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 13:19:06,282 INFO: Epoch:[013/200]
    2022-05-11 13:19:06,283 INFO: Train Loss:0.885 | Acc:0.8460 | F1:0.1553
    2022-05-11 13:19:16,876 INFO: val Loss:0.724 | Acc:0.8491 | F1:0.1561
    2022-05-11 13:19:18,699 INFO: -----------------SAVE:13epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 13:21:17,471 INFO: Epoch:[014/200]
    2022-05-11 13:21:17,471 INFO: Train Loss:0.801 | Acc:0.8486 | F1:0.1581
    2022-05-11 13:21:28,080 INFO: val Loss:0.675 | Acc:0.8515 | F1:0.1677
    2022-05-11 13:21:29,994 INFO: -----------------SAVE:14epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 13:23:28,792 INFO: Epoch:[015/200]
    2022-05-11 13:23:28,792 INFO: Train Loss:0.732 | Acc:0.8498 | F1:0.1715
    2022-05-11 13:23:39,316 INFO: val Loss:0.619 | Acc:0.8585 | F1:0.2054
    2022-05-11 13:23:41,290 INFO: -----------------SAVE:15epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 13:25:40,082 INFO: Epoch:[016/200]
    2022-05-11 13:25:40,082 INFO: Train Loss:0.684 | Acc:0.8539 | F1:0.2003
    2022-05-11 13:25:50,692 INFO: val Loss:0.561 | Acc:0.8643 | F1:0.2374
    2022-05-11 13:25:52,632 INFO: -----------------SAVE:16epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 13:27:51,490 INFO: Epoch:[017/200]
    2022-05-11 13:27:51,490 INFO: Train Loss:0.624 | Acc:0.8618 | F1:0.2475
    2022-05-11 13:28:02,175 INFO: val Loss:0.515 | Acc:0.8772 | F1:0.2877
    2022-05-11 13:28:04,338 INFO: -----------------SAVE:17epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 13:30:03,124 INFO: Epoch:[018/200]
    2022-05-11 13:30:03,125 INFO: Train Loss:0.566 | Acc:0.8705 | F1:0.3040
    2022-05-11 13:30:13,663 INFO: val Loss:0.466 | Acc:0.8865 | F1:0.3385
    2022-05-11 13:30:15,521 INFO: -----------------SAVE:18epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 13:32:14,306 INFO: Epoch:[019/200]
    2022-05-11 13:32:14,307 INFO: Train Loss:0.536 | Acc:0.8799 | F1:0.3414
    2022-05-11 13:32:24,917 INFO: val Loss:0.446 | Acc:0.8924 | F1:0.3778
    2022-05-11 13:32:26,915 INFO: -----------------SAVE:19epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 13:34:25,454 INFO: Epoch:[020/200]
    2022-05-11 13:34:25,454 INFO: Train Loss:0.483 | Acc:0.8881 | F1:0.3870
    2022-05-11 13:34:35,990 INFO: val Loss:0.408 | Acc:0.8982 | F1:0.4229
    2022-05-11 13:34:37,885 INFO: -----------------SAVE:20epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 13:36:36,503 INFO: Epoch:[021/200]
    2022-05-11 13:36:36,503 INFO: Train Loss:0.446 | Acc:0.8963 | F1:0.4321
    2022-05-11 13:36:47,054 INFO: val Loss:0.384 | Acc:0.9006 | F1:0.4281
    2022-05-11 13:36:49,157 INFO: -----------------SAVE:21epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 13:38:47,979 INFO: Epoch:[022/200]
    2022-05-11 13:38:47,979 INFO: Train Loss:0.413 | Acc:0.9033 | F1:0.4767
    2022-05-11 13:38:58,586 INFO: val Loss:0.352 | Acc:0.9029 | F1:0.4625
    2022-05-11 13:39:00,462 INFO: -----------------SAVE:22epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 13:40:59,167 INFO: Epoch:[023/200]
    2022-05-11 13:40:59,167 INFO: Train Loss:0.393 | Acc:0.9050 | F1:0.4964
    2022-05-11 13:41:09,863 INFO: val Loss:0.324 | Acc:0.9064 | F1:0.4523
    2022-05-11 13:41:11,730 INFO: -----------------SAVE:23epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 13:43:10,174 INFO: Epoch:[024/200]
    2022-05-11 13:43:10,174 INFO: Train Loss:0.373 | Acc:0.9071 | F1:0.5070
    2022-05-11 13:43:20,791 INFO: val Loss:0.284 | Acc:0.9205 | F1:0.5286
    2022-05-11 13:43:22,605 INFO: -----------------SAVE:24epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 13:45:21,187 INFO: Epoch:[025/200]
    2022-05-11 13:45:21,187 INFO: Train Loss:0.349 | Acc:0.9106 | F1:0.5232
    2022-05-11 13:45:31,811 INFO: val Loss:0.275 | Acc:0.9228 | F1:0.5410
    2022-05-11 13:45:33,604 INFO: -----------------SAVE:25epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 13:47:31,880 INFO: Epoch:[026/200]
    2022-05-11 13:47:31,881 INFO: Train Loss:0.327 | Acc:0.9150 | F1:0.5659
    2022-05-11 13:47:42,489 INFO: val Loss:0.291 | Acc:0.9228 | F1:0.5739
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 13:49:40,991 INFO: Epoch:[027/200]
    2022-05-11 13:49:40,991 INFO: Train Loss:0.311 | Acc:0.9158 | F1:0.5757
    2022-05-11 13:49:51,478 INFO: val Loss:0.273 | Acc:0.9322 | F1:0.5993
    2022-05-11 13:49:53,520 INFO: -----------------SAVE:27epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 13:51:52,366 INFO: Epoch:[028/200]
    2022-05-11 13:51:52,366 INFO: Train Loss:0.289 | Acc:0.9275 | F1:0.6351
    2022-05-11 13:52:03,147 INFO: val Loss:0.228 | Acc:0.9333 | F1:0.5708
    2022-05-11 13:52:05,166 INFO: -----------------SAVE:28epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 13:54:04,095 INFO: Epoch:[029/200]
    2022-05-11 13:54:04,095 INFO: Train Loss:0.289 | Acc:0.9240 | F1:0.6281
    2022-05-11 13:54:14,779 INFO: val Loss:0.268 | Acc:0.9275 | F1:0.5874
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 13:56:13,404 INFO: Epoch:[030/200]
    2022-05-11 13:56:13,404 INFO: Train Loss:0.279 | Acc:0.9275 | F1:0.6618
    2022-05-11 13:56:24,012 INFO: val Loss:0.203 | Acc:0.9462 | F1:0.7018
    2022-05-11 13:56:25,881 INFO: -----------------SAVE:30epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 13:58:24,374 INFO: Epoch:[031/200]
    2022-05-11 13:58:24,375 INFO: Train Loss:0.273 | Acc:0.9252 | F1:0.6521
    2022-05-11 13:58:34,930 INFO: val Loss:0.228 | Acc:0.9404 | F1:0.6548
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 14:00:33,665 INFO: Epoch:[032/200]
    2022-05-11 14:00:33,665 INFO: Train Loss:0.239 | Acc:0.9363 | F1:0.7095
    2022-05-11 14:00:44,286 INFO: val Loss:0.183 | Acc:0.9415 | F1:0.6682
    2022-05-11 14:00:46,152 INFO: -----------------SAVE:32epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 14:02:44,745 INFO: Epoch:[033/200]
    2022-05-11 14:02:44,745 INFO: Train Loss:0.241 | Acc:0.9372 | F1:0.7131
    2022-05-11 14:02:55,371 INFO: val Loss:0.189 | Acc:0.9415 | F1:0.6649
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 14:04:54,359 INFO: Epoch:[034/200]
    2022-05-11 14:04:54,359 INFO: Train Loss:0.246 | Acc:0.9351 | F1:0.7042
    2022-05-11 14:05:04,959 INFO: val Loss:0.307 | Acc:0.8936 | F1:0.6661
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 14:07:03,454 INFO: Epoch:[035/200]
    2022-05-11 14:07:03,454 INFO: Train Loss:0.220 | Acc:0.9442 | F1:0.7488
    2022-05-11 14:07:14,067 INFO: val Loss:0.197 | Acc:0.9462 | F1:0.7225
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 14:09:12,361 INFO: Epoch:[036/200]
    2022-05-11 14:09:12,362 INFO: Train Loss:0.228 | Acc:0.9395 | F1:0.7347
    2022-05-11 14:09:22,924 INFO: val Loss:0.275 | Acc:0.9228 | F1:0.6875
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 14:11:21,487 INFO: Epoch:[037/200]
    2022-05-11 14:11:21,487 INFO: Train Loss:0.221 | Acc:0.9401 | F1:0.7433
    2022-05-11 14:11:32,090 INFO: val Loss:0.197 | Acc:0.9439 | F1:0.7032
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 14:13:30,280 INFO: Epoch:[038/200]
    2022-05-11 14:13:30,280 INFO: Train Loss:0.224 | Acc:0.9357 | F1:0.7294
    2022-05-11 14:13:40,924 INFO: val Loss:0.187 | Acc:0.9439 | F1:0.6917
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 14:15:39,650 INFO: Epoch:[039/200]
    2022-05-11 14:15:39,651 INFO: Train Loss:0.236 | Acc:0.9372 | F1:0.7400
    2022-05-11 14:15:50,277 INFO: val Loss:0.181 | Acc:0.9532 | F1:0.7444
    2022-05-11 14:15:52,240 INFO: -----------------SAVE:39epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:01<00:00,  1.76it/s]
    2022-05-11 14:17:53,657 INFO: Epoch:[040/200]
    2022-05-11 14:17:53,657 INFO: Train Loss:0.220 | Acc:0.9407 | F1:0.7462
    2022-05-11 14:18:05,296 INFO: val Loss:0.181 | Acc:0.9509 | F1:0.7258
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-11 14:20:04,789 INFO: Epoch:[041/200]
    2022-05-11 14:20:04,789 INFO: Train Loss:0.197 | Acc:0.9486 | F1:0.7808
    2022-05-11 14:20:15,302 INFO: val Loss:0.202 | Acc:0.9520 | F1:0.7292
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.78it/s]
    2022-05-11 14:22:15,755 INFO: Epoch:[042/200]
    2022-05-11 14:22:15,755 INFO: Train Loss:0.214 | Acc:0.9442 | F1:0.7660
    2022-05-11 14:22:26,380 INFO: val Loss:0.227 | Acc:0.9322 | F1:0.6772
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.80it/s]
    2022-05-11 14:24:25,456 INFO: Epoch:[043/200]
    2022-05-11 14:24:25,456 INFO: Train Loss:0.203 | Acc:0.9451 | F1:0.7719
    2022-05-11 14:24:36,345 INFO: val Loss:0.182 | Acc:0.9427 | F1:0.6727
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-11 14:26:35,658 INFO: Epoch:[044/200]
    2022-05-11 14:26:35,658 INFO: Train Loss:0.199 | Acc:0.9483 | F1:0.7736
    2022-05-11 14:26:46,318 INFO: val Loss:0.203 | Acc:0.9520 | F1:0.7138
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:01<00:00,  1.75it/s]
    2022-05-11 14:28:48,278 INFO: Epoch:[045/200]
    2022-05-11 14:28:48,278 INFO: Train Loss:0.196 | Acc:0.9474 | F1:0.7743
    2022-05-11 14:28:58,907 INFO: val Loss:0.189 | Acc:0.9474 | F1:0.7080
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.80it/s]
    2022-05-11 14:30:57,993 INFO: Epoch:[046/200]
    2022-05-11 14:30:57,993 INFO: Train Loss:0.188 | Acc:0.9494 | F1:0.7893
    2022-05-11 14:31:08,693 INFO: val Loss:0.173 | Acc:0.9614 | F1:0.7810
    2022-05-11 14:31:10,460 INFO: -----------------SAVE:46epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-11 14:33:10,078 INFO: Epoch:[047/200]
    2022-05-11 14:33:10,079 INFO: Train Loss:0.193 | Acc:0.9486 | F1:0.7811
    2022-05-11 14:33:20,739 INFO: val Loss:0.198 | Acc:0.9532 | F1:0.7537
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-11 14:35:20,067 INFO: Epoch:[048/200]
    2022-05-11 14:35:20,067 INFO: Train Loss:0.203 | Acc:0.9456 | F1:0.7874
    2022-05-11 14:35:30,872 INFO: val Loss:0.158 | Acc:0.9485 | F1:0.7206
    2022-05-11 14:35:32,871 INFO: -----------------SAVE:48epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:02<00:00,  1.75it/s]
    2022-05-11 14:37:35,149 INFO: Epoch:[049/200]
    2022-05-11 14:37:35,149 INFO: Train Loss:0.186 | Acc:0.9500 | F1:0.7969
    2022-05-11 14:37:46,385 INFO: val Loss:0.154 | Acc:0.9520 | F1:0.8075
    2022-05-11 14:37:48,283 INFO: -----------------SAVE:49epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-11 14:39:47,717 INFO: Epoch:[050/200]
    2022-05-11 14:39:47,717 INFO: Train Loss:0.184 | Acc:0.9503 | F1:0.7945
    2022-05-11 14:39:58,399 INFO: val Loss:0.143 | Acc:0.9602 | F1:0.7619
    2022-05-11 14:40:00,289 INFO: -----------------SAVE:50epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.78it/s]
    2022-05-11 14:42:00,665 INFO: Epoch:[051/200]
    2022-05-11 14:42:00,666 INFO: Train Loss:0.194 | Acc:0.9483 | F1:0.7841
    2022-05-11 14:42:11,277 INFO: val Loss:0.143 | Acc:0.9497 | F1:0.7197
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.78it/s]
    2022-05-11 14:44:11,709 INFO: Epoch:[052/200]
    2022-05-11 14:44:11,710 INFO: Train Loss:0.184 | Acc:0.9544 | F1:0.8168
    2022-05-11 14:44:22,298 INFO: val Loss:0.132 | Acc:0.9649 | F1:0.8067
    2022-05-11 14:44:24,115 INFO: -----------------SAVE:52epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 14:46:22,670 INFO: Epoch:[053/200]
    2022-05-11 14:46:22,670 INFO: Train Loss:0.188 | Acc:0.9512 | F1:0.7991
    2022-05-11 14:46:33,214 INFO: val Loss:0.176 | Acc:0.9450 | F1:0.7744
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 14:48:32,136 INFO: Epoch:[054/200]
    2022-05-11 14:48:32,136 INFO: Train Loss:0.178 | Acc:0.9518 | F1:0.8142
    2022-05-11 14:48:42,730 INFO: val Loss:0.162 | Acc:0.9567 | F1:0.7547
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 14:50:41,551 INFO: Epoch:[055/200]
    2022-05-11 14:50:41,551 INFO: Train Loss:0.181 | Acc:0.9518 | F1:0.8031
    2022-05-11 14:50:52,114 INFO: val Loss:0.152 | Acc:0.9579 | F1:0.7592
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 14:52:50,864 INFO: Epoch:[056/200]
    2022-05-11 14:52:50,864 INFO: Train Loss:0.183 | Acc:0.9521 | F1:0.8127
    2022-05-11 14:53:01,577 INFO: val Loss:0.151 | Acc:0.9556 | F1:0.7613
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-11 14:55:01,358 INFO: Epoch:[057/200]
    2022-05-11 14:55:01,359 INFO: Train Loss:0.188 | Acc:0.9541 | F1:0.8187
    2022-05-11 14:55:11,931 INFO: val Loss:0.171 | Acc:0.9532 | F1:0.7309
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 14:57:10,181 INFO: Epoch:[058/200]
    2022-05-11 14:57:10,181 INFO: Train Loss:0.161 | Acc:0.9553 | F1:0.8139
    2022-05-11 14:57:20,631 INFO: val Loss:0.182 | Acc:0.9544 | F1:0.7436
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 14:59:18,791 INFO: Epoch:[059/200]
    2022-05-11 14:59:18,792 INFO: Train Loss:0.160 | Acc:0.9544 | F1:0.8219
    2022-05-11 14:59:29,314 INFO: val Loss:0.171 | Acc:0.9509 | F1:0.7313
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 15:01:27,465 INFO: Epoch:[060/200]
    2022-05-11 15:01:27,466 INFO: Train Loss:0.178 | Acc:0.9553 | F1:0.8295
    2022-05-11 15:01:38,704 INFO: val Loss:0.163 | Acc:0.9556 | F1:0.7695
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:02<00:00,  1.75it/s]
    2022-05-11 15:03:40,931 INFO: Epoch:[061/200]
    2022-05-11 15:03:40,932 INFO: Train Loss:0.168 | Acc:0.9582 | F1:0.8404
    2022-05-11 15:03:51,811 INFO: val Loss:0.172 | Acc:0.9579 | F1:0.7505
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-11 15:05:51,585 INFO: Epoch:[062/200]
    2022-05-11 15:05:51,586 INFO: Train Loss:0.161 | Acc:0.9562 | F1:0.8274
    2022-05-11 15:06:02,114 INFO: val Loss:0.234 | Acc:0.9474 | F1:0.7607
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 15:08:00,447 INFO: Epoch:[063/200]
    2022-05-11 15:08:00,448 INFO: Train Loss:0.161 | Acc:0.9582 | F1:0.8394
    2022-05-11 15:08:11,049 INFO: val Loss:0.172 | Acc:0.9567 | F1:0.7839
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 15:10:09,175 INFO: Epoch:[064/200]
    2022-05-11 15:10:09,175 INFO: Train Loss:0.154 | Acc:0.9608 | F1:0.8447
    2022-05-11 15:10:19,659 INFO: val Loss:0.172 | Acc:0.9602 | F1:0.8058
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-11 15:12:17,515 INFO: Epoch:[065/200]
    2022-05-11 15:12:17,515 INFO: Train Loss:0.151 | Acc:0.9582 | F1:0.8385
    2022-05-11 15:12:27,927 INFO: val Loss:0.141 | Acc:0.9567 | F1:0.8099
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-11 15:14:25,701 INFO: Epoch:[066/200]
    2022-05-11 15:14:25,702 INFO: Train Loss:0.150 | Acc:0.9576 | F1:0.8296
    2022-05-11 15:14:36,133 INFO: val Loss:0.181 | Acc:0.9368 | F1:0.8113
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.81it/s]
    2022-05-11 15:16:34,047 INFO: Epoch:[067/200]
    2022-05-11 15:16:34,048 INFO: Train Loss:0.125 | Acc:0.9641 | F1:0.8558
    2022-05-11 15:16:44,459 INFO: val Loss:0.164 | Acc:0.9532 | F1:0.7387
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:05<00:00,  1.70it/s]
    2022-05-11 15:18:50,365 INFO: Epoch:[068/200]
    2022-05-11 15:18:50,365 INFO: Train Loss:0.160 | Acc:0.9559 | F1:0.8362
    2022-05-11 15:19:09,164 INFO: val Loss:0.141 | Acc:0.9684 | F1:0.8023
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:40<00:00,  1.33it/s]
    2022-05-11 15:21:49,654 INFO: Epoch:[069/200]
    2022-05-11 15:21:49,654 INFO: Train Loss:0.151 | Acc:0.9600 | F1:0.8374
    2022-05-11 15:22:08,168 INFO: val Loss:0.185 | Acc:0.9474 | F1:0.8200
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:41<00:00,  1.33it/s]
    2022-05-11 15:24:49,504 INFO: Epoch:[070/200]
    2022-05-11 15:24:49,504 INFO: Train Loss:0.150 | Acc:0.9614 | F1:0.8403
    2022-05-11 15:25:07,450 INFO: val Loss:0.141 | Acc:0.9649 | F1:0.8018
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:43<00:00,  1.31it/s]
    2022-05-11 15:27:50,699 INFO: Epoch:[071/200]
    2022-05-11 15:27:50,700 INFO: Train Loss:0.138 | Acc:0.9641 | F1:0.8627
    2022-05-11 15:28:08,999 INFO: val Loss:0.166 | Acc:0.9579 | F1:0.8006
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [05:11<00:00,  1.46s/it]
    2022-05-11 15:33:20,889 INFO: Epoch:[072/200]
    2022-05-11 15:33:20,889 INFO: Train Loss:0.142 | Acc:0.9585 | F1:0.8455
    2022-05-11 15:33:59,168 INFO: val Loss:0.100 | Acc:0.9637 | F1:0.8114
    2022-05-11 15:34:01,036 INFO: -----------------SAVE:72epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [03:13<00:00,  1.11it/s]
    2022-05-11 15:37:14,117 INFO: Epoch:[073/200]
    2022-05-11 15:37:14,118 INFO: Train Loss:0.132 | Acc:0.9649 | F1:0.8645
    2022-05-11 15:37:35,060 INFO: val Loss:0.145 | Acc:0.9544 | F1:0.8020
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:48<00:00,  1.27it/s]
    2022-05-11 15:40:23,580 INFO: Epoch:[074/200]
    2022-05-11 15:40:23,581 INFO: Train Loss:0.133 | Acc:0.9623 | F1:0.8585
    2022-05-11 15:40:41,780 INFO: val Loss:0.152 | Acc:0.9579 | F1:0.7950
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:47<00:00,  1.28it/s]
    2022-05-11 15:43:29,624 INFO: Epoch:[075/200]
    2022-05-11 15:43:29,625 INFO: Train Loss:0.125 | Acc:0.9690 | F1:0.8805
    2022-05-11 15:43:48,398 INFO: val Loss:0.149 | Acc:0.9684 | F1:0.8146
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:50<00:00,  1.25it/s]
    2022-05-11 15:46:39,066 INFO: Epoch:[076/200]
    2022-05-11 15:46:39,066 INFO: Train Loss:0.142 | Acc:0.9643 | F1:0.8635
    2022-05-11 15:46:59,348 INFO: val Loss:0.157 | Acc:0.9579 | F1:0.7737
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:49<00:00,  1.26it/s]
    2022-05-11 15:49:49,050 INFO: Epoch:[077/200]
    2022-05-11 15:49:49,050 INFO: Train Loss:0.120 | Acc:0.9661 | F1:0.8748
    2022-05-11 15:50:09,843 INFO: val Loss:0.145 | Acc:0.9567 | F1:0.7937
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:45<00:00,  1.29it/s]
    2022-05-11 15:52:55,757 INFO: Epoch:[078/200]
    2022-05-11 15:52:55,758 INFO: Train Loss:0.139 | Acc:0.9608 | F1:0.8529
    2022-05-11 15:53:15,148 INFO: val Loss:0.132 | Acc:0.9637 | F1:0.8178
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:44<00:00,  1.30it/s]
    2022-05-11 15:55:59,696 INFO: Epoch:[079/200]
    2022-05-11 15:55:59,696 INFO: Train Loss:0.129 | Acc:0.9667 | F1:0.8732
    2022-05-11 15:56:19,097 INFO: val Loss:0.185 | Acc:0.9380 | F1:0.7969
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [03:04<00:00,  1.16it/s]
    2022-05-11 15:59:23,390 INFO: Epoch:[080/200]
    2022-05-11 15:59:23,391 INFO: Train Loss:0.123 | Acc:0.9658 | F1:0.8775
    2022-05-11 15:59:34,111 INFO: val Loss:0.142 | Acc:0.9544 | F1:0.7698
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-11 16:01:31,470 INFO: Epoch:[081/200]
    2022-05-11 16:01:31,471 INFO: Train Loss:0.116 | Acc:0.9693 | F1:0.8842
    2022-05-11 16:01:42,454 INFO: val Loss:0.215 | Acc:0.9497 | F1:0.7578
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.77it/s]
    2022-05-11 16:03:43,101 INFO: Epoch:[082/200]
    2022-05-11 16:03:43,102 INFO: Train Loss:0.127 | Acc:0.9643 | F1:0.8723
    2022-05-11 16:03:53,877 INFO: val Loss:0.177 | Acc:0.9579 | F1:0.7690
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.78it/s]
    2022-05-11 16:05:54,364 INFO: Epoch:[083/200]
    2022-05-11 16:05:54,365 INFO: Train Loss:0.112 | Acc:0.9693 | F1:0.8906
    2022-05-11 16:06:05,772 INFO: val Loss:0.172 | Acc:0.9579 | F1:0.7982
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-11 16:08:05,553 INFO: Epoch:[084/200]
    2022-05-11 16:08:05,554 INFO: Train Loss:0.131 | Acc:0.9635 | F1:0.8628
    2022-05-11 16:08:16,255 INFO: val Loss:0.172 | Acc:0.9462 | F1:0.7562
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 16:10:15,146 INFO: Epoch:[085/200]
    2022-05-11 16:10:15,147 INFO: Train Loss:0.115 | Acc:0.9705 | F1:0.8825
    2022-05-11 16:10:25,682 INFO: val Loss:0.146 | Acc:0.9649 | F1:0.7971
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 16:12:23,847 INFO: Epoch:[086/200]
    2022-05-11 16:12:23,847 INFO: Train Loss:0.129 | Acc:0.9620 | F1:0.8508
    2022-05-11 16:12:34,517 INFO: val Loss:0.166 | Acc:0.9520 | F1:0.7695
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 16:14:32,574 INFO: Epoch:[087/200]
    2022-05-11 16:14:32,574 INFO: Train Loss:0.124 | Acc:0.9667 | F1:0.8755
    2022-05-11 16:14:43,257 INFO: val Loss:0.175 | Acc:0.9520 | F1:0.7944
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 16:16:42,065 INFO: Epoch:[088/200]
    2022-05-11 16:16:42,066 INFO: Train Loss:0.108 | Acc:0.9699 | F1:0.8857
    2022-05-11 16:16:52,590 INFO: val Loss:0.147 | Acc:0.9567 | F1:0.7978
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-11 16:18:49,874 INFO: Epoch:[089/200]
    2022-05-11 16:18:49,875 INFO: Train Loss:0.116 | Acc:0.9693 | F1:0.8802
    2022-05-11 16:19:00,441 INFO: val Loss:0.172 | Acc:0.9520 | F1:0.7735
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-11 16:20:59,832 INFO: Epoch:[090/200]
    2022-05-11 16:20:59,832 INFO: Train Loss:0.107 | Acc:0.9693 | F1:0.8742
    2022-05-11 16:21:10,541 INFO: val Loss:0.137 | Acc:0.9602 | F1:0.7714
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-11 16:23:08,373 INFO: Epoch:[091/200]
    2022-05-11 16:23:08,373 INFO: Train Loss:0.112 | Acc:0.9717 | F1:0.8900
    2022-05-11 16:23:18,876 INFO: val Loss:0.115 | Acc:0.9778 | F1:0.8982
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-11 16:25:16,161 INFO: Epoch:[092/200]
    2022-05-11 16:25:16,162 INFO: Train Loss:0.090 | Acc:0.9781 | F1:0.9223
    2022-05-11 16:25:26,608 INFO: val Loss:0.124 | Acc:0.9673 | F1:0.8263
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.78it/s]
    2022-05-11 16:27:26,631 INFO: Epoch:[093/200]
    2022-05-11 16:27:26,632 INFO: Train Loss:0.116 | Acc:0.9696 | F1:0.8888
    2022-05-11 16:27:37,879 INFO: val Loss:0.201 | Acc:0.9509 | F1:0.8117
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:01<00:00,  1.76it/s]
    2022-05-11 16:29:39,520 INFO: Epoch:[094/200]
    2022-05-11 16:29:39,520 INFO: Train Loss:0.113 | Acc:0.9696 | F1:0.8816
    2022-05-11 16:29:50,079 INFO: val Loss:0.206 | Acc:0.9520 | F1:0.7743
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:02<00:00,  1.74it/s]
    2022-05-11 16:31:52,910 INFO: Epoch:[095/200]
    2022-05-11 16:31:52,910 INFO: Train Loss:0.110 | Acc:0.9696 | F1:0.8867
    2022-05-11 16:32:03,883 INFO: val Loss:0.164 | Acc:0.9567 | F1:0.7612
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-11 16:34:03,770 INFO: Epoch:[096/200]
    2022-05-11 16:34:03,770 INFO: Train Loss:0.092 | Acc:0.9731 | F1:0.8961
    2022-05-11 16:34:14,459 INFO: val Loss:0.143 | Acc:0.9649 | F1:0.8169
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-11 16:36:13,756 INFO: Epoch:[097/200]
    2022-05-11 16:36:13,756 INFO: Train Loss:0.104 | Acc:0.9749 | F1:0.9080
    2022-05-11 16:36:25,237 INFO: val Loss:0.109 | Acc:0.9684 | F1:0.8361
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-11 16:38:24,830 INFO: Epoch:[098/200]
    2022-05-11 16:38:24,831 INFO: Train Loss:0.107 | Acc:0.9714 | F1:0.8901
    2022-05-11 16:38:35,436 INFO: val Loss:0.136 | Acc:0.9626 | F1:0.8125
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-11 16:40:32,863 INFO: Epoch:[099/200]
    2022-05-11 16:40:32,864 INFO: Train Loss:0.102 | Acc:0.9705 | F1:0.8900
    2022-05-11 16:40:43,353 INFO: val Loss:0.149 | Acc:0.9626 | F1:0.7902
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.81it/s]
    2022-05-11 16:42:41,358 INFO: Epoch:[100/200]
    2022-05-11 16:42:41,359 INFO: Train Loss:0.095 | Acc:0.9743 | F1:0.9006
    2022-05-11 16:42:51,924 INFO: val Loss:0.093 | Acc:0.9684 | F1:0.8220
    2022-05-11 16:42:53,840 INFO: -----------------SAVE:100epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 16:44:52,785 INFO: Epoch:[101/200]
    2022-05-11 16:44:52,785 INFO: Train Loss:0.092 | Acc:0.9737 | F1:0.9006
    2022-05-11 16:45:03,422 INFO: val Loss:0.254 | Acc:0.9310 | F1:0.7795
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.80it/s]
    2022-05-11 16:47:02,585 INFO: Epoch:[102/200]
    2022-05-11 16:47:02,586 INFO: Train Loss:0.094 | Acc:0.9699 | F1:0.8865
    2022-05-11 16:47:13,724 INFO: val Loss:0.144 | Acc:0.9637 | F1:0.8204
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 16:49:12,569 INFO: Epoch:[103/200]
    2022-05-11 16:49:12,570 INFO: Train Loss:0.092 | Acc:0.9766 | F1:0.9173
    2022-05-11 16:49:24,077 INFO: val Loss:0.124 | Acc:0.9696 | F1:0.8416
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-11 16:51:23,587 INFO: Epoch:[104/200]
    2022-05-11 16:51:23,588 INFO: Train Loss:0.093 | Acc:0.9760 | F1:0.9113
    2022-05-11 16:51:34,162 INFO: val Loss:0.134 | Acc:0.9731 | F1:0.8593
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-11 16:53:31,925 INFO: Epoch:[105/200]
    2022-05-11 16:53:31,926 INFO: Train Loss:0.093 | Acc:0.9734 | F1:0.9013
    2022-05-11 16:53:42,488 INFO: val Loss:0.138 | Acc:0.9649 | F1:0.8145
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 16:55:40,899 INFO: Epoch:[106/200]
    2022-05-11 16:55:40,900 INFO: Train Loss:0.079 | Acc:0.9784 | F1:0.9272
    2022-05-11 16:55:51,927 INFO: val Loss:0.135 | Acc:0.9731 | F1:0.8887
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:03<00:00,  1.74it/s]
    2022-05-11 16:57:55,071 INFO: Epoch:[107/200]
    2022-05-11 16:57:55,071 INFO: Train Loss:0.084 | Acc:0.9769 | F1:0.9224
    2022-05-11 16:58:06,260 INFO: val Loss:0.115 | Acc:0.9743 | F1:0.8932
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.78it/s]
    2022-05-11 17:00:06,729 INFO: Epoch:[108/200]
    2022-05-11 17:00:06,729 INFO: Train Loss:0.089 | Acc:0.9746 | F1:0.9063
    2022-05-11 17:00:17,197 INFO: val Loss:0.111 | Acc:0.9731 | F1:0.8617
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-11 17:02:14,820 INFO: Epoch:[109/200]
    2022-05-11 17:02:14,821 INFO: Train Loss:0.103 | Acc:0.9737 | F1:0.9106
    2022-05-11 17:02:25,505 INFO: val Loss:0.120 | Acc:0.9766 | F1:0.8917
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 17:04:23,581 INFO: Epoch:[110/200]
    2022-05-11 17:04:23,581 INFO: Train Loss:0.077 | Acc:0.9781 | F1:0.9224
    2022-05-11 17:04:34,102 INFO: val Loss:0.131 | Acc:0.9731 | F1:0.8842
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.78it/s]
    2022-05-11 17:06:34,200 INFO: Epoch:[111/200]
    2022-05-11 17:06:34,201 INFO: Train Loss:0.075 | Acc:0.9778 | F1:0.9192
    2022-05-11 17:06:44,904 INFO: val Loss:0.165 | Acc:0.9567 | F1:0.8298
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 17:08:43,466 INFO: Epoch:[112/200]
    2022-05-11 17:08:43,466 INFO: Train Loss:0.080 | Acc:0.9755 | F1:0.9091
    2022-05-11 17:08:54,165 INFO: val Loss:0.139 | Acc:0.9684 | F1:0.8377
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 17:10:52,977 INFO: Epoch:[113/200]
    2022-05-11 17:10:52,977 INFO: Train Loss:0.086 | Acc:0.9778 | F1:0.9215
    2022-05-11 17:11:03,779 INFO: val Loss:0.148 | Acc:0.9684 | F1:0.8479
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-11 17:13:03,618 INFO: Epoch:[114/200]
    2022-05-11 17:13:03,619 INFO: Train Loss:0.078 | Acc:0.9784 | F1:0.9239
    2022-05-11 17:13:14,499 INFO: val Loss:0.166 | Acc:0.9626 | F1:0.8200
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [05:21<00:00,  1.50s/it]
    2022-05-11 17:18:36,250 INFO: Epoch:[115/200]
    2022-05-11 17:18:36,251 INFO: Train Loss:0.089 | Acc:0.9775 | F1:0.9285
    2022-05-11 17:19:33,067 INFO: val Loss:0.180 | Acc:0.9614 | F1:0.8442
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [07:29<00:00,  2.10s/it]
    2022-05-11 17:27:02,460 INFO: Epoch:[116/200]
    2022-05-11 17:27:02,461 INFO: Train Loss:0.093 | Acc:0.9755 | F1:0.9079
    2022-05-11 17:27:56,038 INFO: val Loss:0.125 | Acc:0.9684 | F1:0.8684
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [07:02<00:00,  1.98s/it]
    2022-05-11 17:34:58,908 INFO: Epoch:[117/200]
    2022-05-11 17:34:58,908 INFO: Train Loss:0.080 | Acc:0.9772 | F1:0.9181
    2022-05-11 17:35:49,352 INFO: val Loss:0.168 | Acc:0.9556 | F1:0.8119
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [06:29<00:00,  1.82s/it]
    2022-05-11 17:42:19,155 INFO: Epoch:[118/200]
    2022-05-11 17:42:19,155 INFO: Train Loss:0.073 | Acc:0.9798 | F1:0.9324
    2022-05-11 17:43:02,327 INFO: val Loss:0.127 | Acc:0.9708 | F1:0.8806
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [06:21<00:00,  1.78s/it]
    2022-05-11 17:49:24,238 INFO: Epoch:[119/200]
    2022-05-11 17:49:24,238 INFO: Train Loss:0.070 | Acc:0.9813 | F1:0.9360
    2022-05-11 17:49:35,172 INFO: val Loss:0.105 | Acc:0.9754 | F1:0.8843
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:01<00:00,  1.77it/s]
    2022-05-11 17:51:36,282 INFO: Epoch:[120/200]
    2022-05-11 17:51:36,283 INFO: Train Loss:0.076 | Acc:0.9810 | F1:0.9316
    2022-05-11 17:51:46,803 INFO: val Loss:0.098 | Acc:0.9696 | F1:0.8619
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.81it/s]
    2022-05-11 17:53:44,729 INFO: Epoch:[121/200]
    2022-05-11 17:53:44,730 INFO: Train Loss:0.062 | Acc:0.9801 | F1:0.9196
    2022-05-11 17:53:55,280 INFO: val Loss:0.145 | Acc:0.9637 | F1:0.8550
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.78it/s]
    2022-05-11 17:55:55,312 INFO: Epoch:[122/200]
    2022-05-11 17:55:55,313 INFO: Train Loss:0.070 | Acc:0.9810 | F1:0.9309
    2022-05-11 17:56:05,923 INFO: val Loss:0.135 | Acc:0.9637 | F1:0.8221
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.78it/s]
    2022-05-11 17:58:06,333 INFO: Epoch:[123/200]
    2022-05-11 17:58:06,333 INFO: Train Loss:0.070 | Acc:0.9804 | F1:0.9330
    2022-05-11 17:58:17,043 INFO: val Loss:0.133 | Acc:0.9661 | F1:0.8475
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 18:00:15,266 INFO: Epoch:[124/200]
    2022-05-11 18:00:15,266 INFO: Train Loss:0.068 | Acc:0.9816 | F1:0.9430
    2022-05-11 18:00:25,779 INFO: val Loss:0.103 | Acc:0.9743 | F1:0.8813
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-11 18:02:25,218 INFO: Epoch:[125/200]
    2022-05-11 18:02:25,218 INFO: Train Loss:0.062 | Acc:0.9819 | F1:0.9347
    2022-05-11 18:02:35,832 INFO: val Loss:0.129 | Acc:0.9731 | F1:0.8796
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 18:04:33,979 INFO: Epoch:[126/200]
    2022-05-11 18:04:33,979 INFO: Train Loss:0.069 | Acc:0.9801 | F1:0.9285
    2022-05-11 18:04:44,654 INFO: val Loss:0.118 | Acc:0.9684 | F1:0.8583
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-11 18:06:41,634 INFO: Epoch:[127/200]
    2022-05-11 18:06:41,634 INFO: Train Loss:0.068 | Acc:0.9798 | F1:0.9314
    2022-05-11 18:06:52,341 INFO: val Loss:0.116 | Acc:0.9719 | F1:0.8821
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-11 18:08:52,177 INFO: Epoch:[128/200]
    2022-05-11 18:08:52,177 INFO: Train Loss:0.053 | Acc:0.9857 | F1:0.9522
    2022-05-11 18:09:03,005 INFO: val Loss:0.133 | Acc:0.9731 | F1:0.8791
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.78it/s]
    2022-05-11 18:11:03,090 INFO: Epoch:[129/200]
    2022-05-11 18:11:03,091 INFO: Train Loss:0.063 | Acc:0.9868 | F1:0.9504
    2022-05-11 18:11:13,555 INFO: val Loss:0.117 | Acc:0.9696 | F1:0.8457
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.78it/s]
    2022-05-11 18:13:13,990 INFO: Epoch:[130/200]
    2022-05-11 18:13:13,990 INFO: Train Loss:0.075 | Acc:0.9801 | F1:0.9291
    2022-05-11 18:13:24,682 INFO: val Loss:0.122 | Acc:0.9708 | F1:0.8688
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 18:15:23,110 INFO: Epoch:[131/200]
    2022-05-11 18:15:23,110 INFO: Train Loss:0.050 | Acc:0.9871 | F1:0.9565
    2022-05-11 18:15:33,670 INFO: val Loss:0.111 | Acc:0.9754 | F1:0.8966
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 18:17:31,999 INFO: Epoch:[132/200]
    2022-05-11 18:17:31,999 INFO: Train Loss:0.058 | Acc:0.9825 | F1:0.9395
    2022-05-11 18:17:42,798 INFO: val Loss:0.102 | Acc:0.9731 | F1:0.8762
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-11 18:19:42,063 INFO: Epoch:[133/200]
    2022-05-11 18:19:42,063 INFO: Train Loss:0.059 | Acc:0.9825 | F1:0.9400
    2022-05-11 18:19:52,630 INFO: val Loss:0.083 | Acc:0.9813 | F1:0.9120
    2022-05-11 18:19:54,525 INFO: -----------------SAVE:133epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 18:21:53,426 INFO: Epoch:[134/200]
    2022-05-11 18:21:53,427 INFO: Train Loss:0.053 | Acc:0.9848 | F1:0.9503
    2022-05-11 18:22:05,065 INFO: val Loss:0.120 | Acc:0.9743 | F1:0.8576
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:05<00:00,  1.70it/s]
    2022-05-11 18:24:10,699 INFO: Epoch:[135/200]
    2022-05-11 18:24:10,699 INFO: Train Loss:0.053 | Acc:0.9851 | F1:0.9554
    2022-05-11 18:24:22,363 INFO: val Loss:0.084 | Acc:0.9813 | F1:0.9095
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:05<00:00,  1.71it/s]
    2022-05-11 18:26:27,613 INFO: Epoch:[136/200]
    2022-05-11 18:26:27,613 INFO: Train Loss:0.056 | Acc:0.9836 | F1:0.9465
    2022-05-11 18:26:39,414 INFO: val Loss:0.139 | Acc:0.9661 | F1:0.8275
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:05<00:00,  1.71it/s]
    2022-05-11 18:28:44,734 INFO: Epoch:[137/200]
    2022-05-11 18:28:44,734 INFO: Train Loss:0.057 | Acc:0.9839 | F1:0.9499
    2022-05-11 18:28:56,441 INFO: val Loss:0.157 | Acc:0.9684 | F1:0.8478
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:03<00:00,  1.73it/s]
    2022-05-11 18:31:00,218 INFO: Epoch:[138/200]
    2022-05-11 18:31:00,218 INFO: Train Loss:0.070 | Acc:0.9839 | F1:0.9433
    2022-05-11 18:31:10,879 INFO: val Loss:0.129 | Acc:0.9696 | F1:0.8389
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 18:33:06,825 INFO: Epoch:[139/200]
    2022-05-11 18:33:06,826 INFO: Train Loss:0.045 | Acc:0.9863 | F1:0.9539
    2022-05-11 18:33:17,202 INFO: val Loss:0.118 | Acc:0.9731 | F1:0.8800
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 18:35:13,243 INFO: Epoch:[140/200]
    2022-05-11 18:35:13,243 INFO: Train Loss:0.050 | Acc:0.9845 | F1:0.9446
    2022-05-11 18:35:23,646 INFO: val Loss:0.109 | Acc:0.9719 | F1:0.8543
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 18:37:19,582 INFO: Epoch:[141/200]
    2022-05-11 18:37:19,583 INFO: Train Loss:0.052 | Acc:0.9845 | F1:0.9526
    2022-05-11 18:37:29,952 INFO: val Loss:0.220 | Acc:0.9567 | F1:0.8874
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 18:39:25,700 INFO: Epoch:[142/200]
    2022-05-11 18:39:25,701 INFO: Train Loss:0.047 | Acc:0.9874 | F1:0.9651
    2022-05-11 18:39:36,092 INFO: val Loss:0.115 | Acc:0.9696 | F1:0.8770
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 18:41:32,102 INFO: Epoch:[143/200]
    2022-05-11 18:41:32,103 INFO: Train Loss:0.044 | Acc:0.9880 | F1:0.9642
    2022-05-11 18:41:42,535 INFO: val Loss:0.140 | Acc:0.9696 | F1:0.8543
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 18:43:38,368 INFO: Epoch:[144/200]
    2022-05-11 18:43:38,368 INFO: Train Loss:0.047 | Acc:0.9886 | F1:0.9596
    2022-05-11 18:43:48,807 INFO: val Loss:0.098 | Acc:0.9754 | F1:0.8824
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 18:45:44,775 INFO: Epoch:[145/200]
    2022-05-11 18:45:44,775 INFO: Train Loss:0.043 | Acc:0.9866 | F1:0.9476
    2022-05-11 18:45:55,187 INFO: val Loss:0.118 | Acc:0.9673 | F1:0.8602
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 18:47:51,180 INFO: Epoch:[146/200]
    2022-05-11 18:47:51,180 INFO: Train Loss:0.038 | Acc:0.9901 | F1:0.9658
    2022-05-11 18:48:01,699 INFO: val Loss:0.111 | Acc:0.9696 | F1:0.8566
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 18:49:57,452 INFO: Epoch:[147/200]
    2022-05-11 18:49:57,452 INFO: Train Loss:0.049 | Acc:0.9868 | F1:0.9589
    2022-05-11 18:50:07,921 INFO: val Loss:0.156 | Acc:0.9649 | F1:0.8374
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 18:52:03,750 INFO: Epoch:[148/200]
    2022-05-11 18:52:03,751 INFO: Train Loss:0.046 | Acc:0.9874 | F1:0.9591
    2022-05-11 18:52:14,174 INFO: val Loss:0.121 | Acc:0.9731 | F1:0.8872
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 18:54:09,984 INFO: Epoch:[149/200]
    2022-05-11 18:54:09,984 INFO: Train Loss:0.040 | Acc:0.9901 | F1:0.9640
    2022-05-11 18:54:20,438 INFO: val Loss:0.138 | Acc:0.9708 | F1:0.8574
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 18:56:16,517 INFO: Epoch:[150/200]
    2022-05-11 18:56:16,518 INFO: Train Loss:0.038 | Acc:0.9895 | F1:0.9617
    2022-05-11 18:56:26,912 INFO: val Loss:0.122 | Acc:0.9684 | F1:0.8642
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 18:58:22,999 INFO: Epoch:[151/200]
    2022-05-11 18:58:22,999 INFO: Train Loss:0.037 | Acc:0.9912 | F1:0.9719
    2022-05-11 18:58:33,423 INFO: val Loss:0.110 | Acc:0.9743 | F1:0.8737
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-11 19:00:30,800 INFO: Epoch:[152/200]
    2022-05-11 19:00:30,801 INFO: Train Loss:0.037 | Acc:0.9880 | F1:0.9603
    2022-05-11 19:00:41,713 INFO: val Loss:0.089 | Acc:0.9801 | F1:0.8866
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 19:02:37,997 INFO: Epoch:[153/200]
    2022-05-11 19:02:37,998 INFO: Train Loss:0.039 | Acc:0.9871 | F1:0.9525
    2022-05-11 19:02:48,450 INFO: val Loss:0.121 | Acc:0.9719 | F1:0.8525
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.83it/s]
    2022-05-11 19:04:45,669 INFO: Epoch:[154/200]
    2022-05-11 19:04:45,669 INFO: Train Loss:0.046 | Acc:0.9898 | F1:0.9685
    2022-05-11 19:04:56,809 INFO: val Loss:0.102 | Acc:0.9789 | F1:0.8887
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 19:06:53,277 INFO: Epoch:[155/200]
    2022-05-11 19:06:53,277 INFO: Train Loss:0.035 | Acc:0.9895 | F1:0.9689
    2022-05-11 19:07:03,760 INFO: val Loss:0.100 | Acc:0.9836 | F1:0.9131
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 19:09:00,016 INFO: Epoch:[156/200]
    2022-05-11 19:09:00,017 INFO: Train Loss:0.039 | Acc:0.9871 | F1:0.9612
    2022-05-11 19:09:10,520 INFO: val Loss:0.092 | Acc:0.9766 | F1:0.8734
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 19:11:06,708 INFO: Epoch:[157/200]
    2022-05-11 19:11:06,709 INFO: Train Loss:0.027 | Acc:0.9924 | F1:0.9718
    2022-05-11 19:11:17,112 INFO: val Loss:0.087 | Acc:0.9743 | F1:0.8522
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 19:13:13,680 INFO: Epoch:[158/200]
    2022-05-11 19:13:13,681 INFO: Train Loss:0.032 | Acc:0.9895 | F1:0.9655
    2022-05-11 19:13:24,125 INFO: val Loss:0.114 | Acc:0.9731 | F1:0.8593
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-11 19:15:20,843 INFO: Epoch:[159/200]
    2022-05-11 19:15:20,844 INFO: Train Loss:0.025 | Acc:0.9921 | F1:0.9782
    2022-05-11 19:15:31,323 INFO: val Loss:0.101 | Acc:0.9789 | F1:0.8920
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 19:17:27,648 INFO: Epoch:[160/200]
    2022-05-11 19:17:27,649 INFO: Train Loss:0.036 | Acc:0.9892 | F1:0.9614
    2022-05-11 19:17:38,148 INFO: val Loss:0.115 | Acc:0.9743 | F1:0.8704
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-11 19:19:34,911 INFO: Epoch:[161/200]
    2022-05-11 19:19:34,911 INFO: Train Loss:0.027 | Acc:0.9930 | F1:0.9711
    2022-05-11 19:19:45,417 INFO: val Loss:0.107 | Acc:0.9766 | F1:0.8646
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-11 19:21:42,199 INFO: Epoch:[162/200]
    2022-05-11 19:21:42,200 INFO: Train Loss:0.030 | Acc:0.9912 | F1:0.9734
    2022-05-11 19:21:52,656 INFO: val Loss:0.092 | Acc:0.9766 | F1:0.8767
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-11 19:23:49,481 INFO: Epoch:[163/200]
    2022-05-11 19:23:49,481 INFO: Train Loss:0.029 | Acc:0.9921 | F1:0.9758
    2022-05-11 19:23:59,945 INFO: val Loss:0.094 | Acc:0.9789 | F1:0.9033
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.78it/s]
    2022-05-11 19:26:00,307 INFO: Epoch:[164/200]
    2022-05-11 19:26:00,308 INFO: Train Loss:0.026 | Acc:0.9924 | F1:0.9746
    2022-05-11 19:26:12,132 INFO: val Loss:0.111 | Acc:0.9778 | F1:0.8832
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:02<00:00,  1.75it/s]
    2022-05-11 19:28:14,364 INFO: Epoch:[165/200]
    2022-05-11 19:28:14,364 INFO: Train Loss:0.034 | Acc:0.9904 | F1:0.9687
    2022-05-11 19:28:25,355 INFO: val Loss:0.101 | Acc:0.9801 | F1:0.9060
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:01<00:00,  1.76it/s]
    2022-05-11 19:30:26,708 INFO: Epoch:[166/200]
    2022-05-11 19:30:26,709 INFO: Train Loss:0.026 | Acc:0.9924 | F1:0.9800
    2022-05-11 19:30:37,512 INFO: val Loss:0.104 | Acc:0.9825 | F1:0.9142
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-11 19:32:36,858 INFO: Epoch:[167/200]
    2022-05-11 19:32:36,858 INFO: Train Loss:0.022 | Acc:0.9927 | F1:0.9767
    2022-05-11 19:32:47,507 INFO: val Loss:0.091 | Acc:0.9836 | F1:0.9058
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.83it/s]
    2022-05-11 19:34:44,609 INFO: Epoch:[168/200]
    2022-05-11 19:34:44,609 INFO: Train Loss:0.030 | Acc:0.9915 | F1:0.9763
    2022-05-11 19:34:55,063 INFO: val Loss:0.088 | Acc:0.9813 | F1:0.9186
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 19:36:53,178 INFO: Epoch:[169/200]
    2022-05-11 19:36:53,179 INFO: Train Loss:0.027 | Acc:0.9939 | F1:0.9820
    2022-05-11 19:37:03,846 INFO: val Loss:0.082 | Acc:0.9825 | F1:0.9174
    2022-05-11 19:37:05,790 INFO: -----------------SAVE:169epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-11 19:39:02,674 INFO: Epoch:[170/200]
    2022-05-11 19:39:02,674 INFO: Train Loss:0.020 | Acc:0.9944 | F1:0.9838
    2022-05-11 19:39:13,118 INFO: val Loss:0.086 | Acc:0.9813 | F1:0.9072
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 19:41:09,679 INFO: Epoch:[171/200]
    2022-05-11 19:41:09,679 INFO: Train Loss:0.026 | Acc:0.9930 | F1:0.9787
    2022-05-11 19:41:20,098 INFO: val Loss:0.070 | Acc:0.9883 | F1:0.9404
    2022-05-11 19:41:21,972 INFO: -----------------SAVE:171epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 19:43:18,129 INFO: Epoch:[172/200]
    2022-05-11 19:43:18,130 INFO: Train Loss:0.027 | Acc:0.9915 | F1:0.9730
    2022-05-11 19:43:28,590 INFO: val Loss:0.083 | Acc:0.9825 | F1:0.9134
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 19:45:24,848 INFO: Epoch:[173/200]
    2022-05-11 19:45:24,849 INFO: Train Loss:0.023 | Acc:0.9930 | F1:0.9774
    2022-05-11 19:45:35,316 INFO: val Loss:0.093 | Acc:0.9813 | F1:0.8989
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 19:47:31,931 INFO: Epoch:[174/200]
    2022-05-11 19:47:31,932 INFO: Train Loss:0.023 | Acc:0.9933 | F1:0.9774
    2022-05-11 19:47:42,349 INFO: val Loss:0.075 | Acc:0.9801 | F1:0.9096
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 19:49:38,718 INFO: Epoch:[175/200]
    2022-05-11 19:49:38,718 INFO: Train Loss:0.021 | Acc:0.9939 | F1:0.9810
    2022-05-11 19:49:49,193 INFO: val Loss:0.073 | Acc:0.9825 | F1:0.9235
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-11 19:51:46,112 INFO: Epoch:[176/200]
    2022-05-11 19:51:46,112 INFO: Train Loss:0.017 | Acc:0.9947 | F1:0.9826
    2022-05-11 19:51:56,555 INFO: val Loss:0.065 | Acc:0.9836 | F1:0.9225
    2022-05-11 19:51:58,443 INFO: -----------------SAVE:176epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:02<00:00,  1.75it/s]
    2022-05-11 19:54:00,458 INFO: Epoch:[177/200]
    2022-05-11 19:54:00,459 INFO: Train Loss:0.020 | Acc:0.9953 | F1:0.9870
    2022-05-11 19:54:11,416 INFO: val Loss:0.095 | Acc:0.9825 | F1:0.9187
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-11 19:56:10,666 INFO: Epoch:[178/200]
    2022-05-11 19:56:10,666 INFO: Train Loss:0.018 | Acc:0.9947 | F1:0.9838
    2022-05-11 19:56:21,126 INFO: val Loss:0.082 | Acc:0.9836 | F1:0.9337
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 19:58:17,341 INFO: Epoch:[179/200]
    2022-05-11 19:58:17,341 INFO: Train Loss:0.020 | Acc:0.9942 | F1:0.9789
    2022-05-11 19:58:27,817 INFO: val Loss:0.073 | Acc:0.9848 | F1:0.9335
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-11 20:00:25,662 INFO: Epoch:[180/200]
    2022-05-11 20:00:25,662 INFO: Train Loss:0.019 | Acc:0.9944 | F1:0.9796
    2022-05-11 20:00:36,932 INFO: val Loss:0.070 | Acc:0.9860 | F1:0.9321
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-11 20:02:34,443 INFO: Epoch:[181/200]
    2022-05-11 20:02:34,444 INFO: Train Loss:0.019 | Acc:0.9930 | F1:0.9745
    2022-05-11 20:02:44,963 INFO: val Loss:0.080 | Acc:0.9860 | F1:0.9261
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-11 20:04:41,956 INFO: Epoch:[182/200]
    2022-05-11 20:04:41,957 INFO: Train Loss:0.019 | Acc:0.9950 | F1:0.9838
    2022-05-11 20:04:52,441 INFO: val Loss:0.080 | Acc:0.9836 | F1:0.9063
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 20:06:49,013 INFO: Epoch:[183/200]
    2022-05-11 20:06:49,013 INFO: Train Loss:0.017 | Acc:0.9953 | F1:0.9868
    2022-05-11 20:06:59,484 INFO: val Loss:0.090 | Acc:0.9825 | F1:0.9040
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 20:08:57,704 INFO: Epoch:[184/200]
    2022-05-11 20:08:57,705 INFO: Train Loss:0.015 | Acc:0.9950 | F1:0.9837
    2022-05-11 20:09:08,160 INFO: val Loss:0.086 | Acc:0.9836 | F1:0.9131
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-11 20:11:04,978 INFO: Epoch:[185/200]
    2022-05-11 20:11:04,978 INFO: Train Loss:0.013 | Acc:0.9953 | F1:0.9821
    2022-05-11 20:11:15,418 INFO: val Loss:0.088 | Acc:0.9825 | F1:0.9066
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.83it/s]
    2022-05-11 20:13:12,563 INFO: Epoch:[186/200]
    2022-05-11 20:13:12,563 INFO: Train Loss:0.011 | Acc:0.9968 | F1:0.9907
    2022-05-11 20:13:23,028 INFO: val Loss:0.086 | Acc:0.9836 | F1:0.9146
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-11 20:15:19,885 INFO: Epoch:[187/200]
    2022-05-11 20:15:19,886 INFO: Train Loss:0.013 | Acc:0.9953 | F1:0.9841
    2022-05-11 20:15:30,312 INFO: val Loss:0.082 | Acc:0.9848 | F1:0.9195
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-11 20:17:27,868 INFO: Epoch:[188/200]
    2022-05-11 20:17:27,868 INFO: Train Loss:0.017 | Acc:0.9956 | F1:0.9868
    2022-05-11 20:17:38,314 INFO: val Loss:0.079 | Acc:0.9836 | F1:0.9146
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-11 20:19:35,727 INFO: Epoch:[189/200]
    2022-05-11 20:19:35,727 INFO: Train Loss:0.011 | Acc:0.9974 | F1:0.9907
    2022-05-11 20:19:46,304 INFO: val Loss:0.081 | Acc:0.9848 | F1:0.9206
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 20:21:44,487 INFO: Epoch:[190/200]
    2022-05-11 20:21:44,487 INFO: Train Loss:0.012 | Acc:0.9962 | F1:0.9883
    2022-05-11 20:21:55,558 INFO: val Loss:0.081 | Acc:0.9836 | F1:0.9146
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 20:23:53,745 INFO: Epoch:[191/200]
    2022-05-11 20:23:53,746 INFO: Train Loss:0.016 | Acc:0.9962 | F1:0.9868
    2022-05-11 20:24:05,433 INFO: val Loss:0.074 | Acc:0.9836 | F1:0.9132
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 20:26:04,338 INFO: Epoch:[192/200]
    2022-05-11 20:26:04,338 INFO: Train Loss:0.013 | Acc:0.9956 | F1:0.9844
    2022-05-11 20:26:15,166 INFO: val Loss:0.079 | Acc:0.9813 | F1:0.9062
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-11 20:28:12,765 INFO: Epoch:[193/200]
    2022-05-11 20:28:12,766 INFO: Train Loss:0.011 | Acc:0.9962 | F1:0.9898
    2022-05-11 20:28:23,167 INFO: val Loss:0.085 | Acc:0.9813 | F1:0.9065
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 20:30:21,806 INFO: Epoch:[194/200]
    2022-05-11 20:30:21,807 INFO: Train Loss:0.012 | Acc:0.9956 | F1:0.9832
    2022-05-11 20:30:32,815 INFO: val Loss:0.082 | Acc:0.9813 | F1:0.9055
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.78it/s]
    2022-05-11 20:32:32,997 INFO: Epoch:[195/200]
    2022-05-11 20:32:32,998 INFO: Train Loss:0.010 | Acc:0.9974 | F1:0.9900
    2022-05-11 20:32:43,356 INFO: val Loss:0.080 | Acc:0.9848 | F1:0.9195
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 20:34:39,304 INFO: Epoch:[196/200]
    2022-05-11 20:34:39,304 INFO: Train Loss:0.010 | Acc:0.9962 | F1:0.9851
    2022-05-11 20:34:49,738 INFO: val Loss:0.075 | Acc:0.9848 | F1:0.9135
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.84it/s]
    2022-05-11 20:36:45,737 INFO: Epoch:[197/200]
    2022-05-11 20:36:45,738 INFO: Train Loss:0.011 | Acc:0.9965 | F1:0.9900
    2022-05-11 20:36:56,160 INFO: val Loss:0.077 | Acc:0.9848 | F1:0.9195
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 20:38:52,194 INFO: Epoch:[198/200]
    2022-05-11 20:38:52,195 INFO: Train Loss:0.011 | Acc:0.9959 | F1:0.9872
    2022-05-11 20:39:02,692 INFO: val Loss:0.081 | Acc:0.9848 | F1:0.9195
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 20:40:58,582 INFO: Epoch:[199/200]
    2022-05-11 20:40:58,583 INFO: Train Loss:0.014 | Acc:0.9962 | F1:0.9894
    2022-05-11 20:41:09,032 INFO: val Loss:0.075 | Acc:0.9848 | F1:0.9195
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 20:43:04,951 INFO: Epoch:[200/200]
    2022-05-11 20:43:04,951 INFO: Train Loss:0.015 | Acc:0.9968 | F1:0.9886
    2022-05-11 20:43:15,354 INFO: val Loss:0.078 | Acc:0.9848 | F1:0.9169
    2022-05-11 20:43:15,355 INFO: 
    Best Val Epoch:176 | Val Loss:0.0652 | Val Acc:0.9836 | Val F1:0.9225
    2022-05-11 20:43:15,355 INFO: Total Process time:472.058Minute
    2022-05-11 20:43:15,358 INFO: {'exp_num': '3', 'data_path': './data', 'Kfold': 5, 'model_path': 'results/', 'encoder_name': 'regnety_160', 'drop_path_rate': 0.2, 'img_size': 224, 'batch_size': 16, 'epochs': 200, 'optimizer': 'Lamb', 'initial_lr': 5e-06, 'weight_decay': 0.001, 'aug_ver': 2, 'scheduler': 'cycle', 'warm_epoch': 5, 'max_lr': 0.001, 'min_lr': 5e-06, 'tmax': 145, 'patience': 50, 'clipping': None, 'amp': True, 'multi_gpu': False, 'logging': False, 'num_workers': 0, 'seed': 42, 'fold': 3}
    

    <---- Training Params ---->
    Read train_df.csv
    Dataset size:3422
    Dataset size:855
    

    2022-05-11 20:43:16,174 INFO: Loading pretrained weights from url (https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth)
    2022-05-11 20:43:16,547 INFO: Computational complexity:       15.93 GMac
    2022-05-11 20:43:16,548 INFO: Number of parameters:           80.83 M 
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 20:45:12,110 INFO: Epoch:[001/200]
    2022-05-11 20:45:12,111 INFO: Train Loss:4.486 | Acc:0.0085 | F1:0.0030
    2022-05-11 20:45:22,613 INFO: val Loss:4.530 | Acc:0.0000 | F1:0.0000
    2022-05-11 20:45:24,585 INFO: -----------------SAVE:1epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.78it/s]
    2022-05-11 20:47:25,132 INFO: Epoch:[002/200]
    2022-05-11 20:47:25,132 INFO: Train Loss:4.476 | Acc:0.0111 | F1:0.0039
    2022-05-11 20:47:36,862 INFO: val Loss:4.519 | Acc:0.0000 | F1:0.0000
    2022-05-11 20:47:38,770 INFO: -----------------SAVE:2epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-11 20:49:38,173 INFO: Epoch:[003/200]
    2022-05-11 20:49:38,174 INFO: Train Loss:4.440 | Acc:0.0167 | F1:0.0049
    2022-05-11 20:49:49,819 INFO: val Loss:4.488 | Acc:0.0035 | F1:0.0005
    2022-05-11 20:49:51,674 INFO: -----------------SAVE:3epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-11 20:51:48,361 INFO: Epoch:[004/200]
    2022-05-11 20:51:48,362 INFO: Train Loss:4.403 | Acc:0.0313 | F1:0.0082
    2022-05-11 20:51:58,993 INFO: val Loss:4.457 | Acc:0.0058 | F1:0.0009
    2022-05-11 20:52:00,900 INFO: -----------------SAVE:4epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 20:53:57,393 INFO: Epoch:[005/200]
    2022-05-11 20:53:57,393 INFO: Train Loss:4.348 | Acc:0.0675 | F1:0.0156
    2022-05-11 20:54:08,089 INFO: val Loss:4.400 | Acc:0.0222 | F1:0.0037
    2022-05-11 20:54:09,928 INFO: -----------------SAVE:5epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 20:56:06,365 INFO: Epoch:[006/200]
    2022-05-11 20:56:06,365 INFO: Train Loss:4.014 | Acc:0.3895 | F1:0.0774
    2022-05-11 20:56:16,781 INFO: val Loss:3.748 | Acc:0.5579 | F1:0.1051
    2022-05-11 20:56:18,690 INFO: -----------------SAVE:6epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.83it/s]
    2022-05-11 20:58:15,818 INFO: Epoch:[007/200]
    2022-05-11 20:58:15,818 INFO: Train Loss:3.216 | Acc:0.7516 | F1:0.1361
    2022-05-11 20:58:26,329 INFO: val Loss:2.740 | Acc:0.7661 | F1:0.1342
    2022-05-11 20:58:28,157 INFO: -----------------SAVE:7epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 21:00:24,257 INFO: Epoch:[008/200]
    2022-05-11 21:00:24,258 INFO: Train Loss:2.350 | Acc:0.8241 | F1:0.1460
    2022-05-11 21:00:34,992 INFO: val Loss:1.822 | Acc:0.8351 | F1:0.1465
    2022-05-11 21:00:36,944 INFO: -----------------SAVE:8epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.83it/s]
    2022-05-11 21:02:34,199 INFO: Epoch:[009/200]
    2022-05-11 21:02:34,199 INFO: Train Loss:1.614 | Acc:0.8326 | F1:0.1484
    2022-05-11 21:02:44,579 INFO: val Loss:1.128 | Acc:0.8433 | F1:0.1536
    2022-05-11 21:02:46,444 INFO: -----------------SAVE:9epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 21:04:42,947 INFO: Epoch:[010/200]
    2022-05-11 21:04:42,948 INFO: Train Loss:1.245 | Acc:0.8396 | F1:0.1527
    2022-05-11 21:04:53,509 INFO: val Loss:0.998 | Acc:0.8480 | F1:0.1561
    2022-05-11 21:04:55,489 INFO: -----------------SAVE:10epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 21:06:51,844 INFO: Epoch:[011/200]
    2022-05-11 21:06:51,844 INFO: Train Loss:1.114 | Acc:0.8463 | F1:0.1556
    2022-05-11 21:07:02,366 INFO: val Loss:0.896 | Acc:0.8480 | F1:0.1561
    2022-05-11 21:07:04,197 INFO: -----------------SAVE:11epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 21:09:00,505 INFO: Epoch:[012/200]
    2022-05-11 21:09:00,506 INFO: Train Loss:1.006 | Acc:0.8469 | F1:0.1558
    2022-05-11 21:09:10,967 INFO: val Loss:0.815 | Acc:0.8480 | F1:0.1561
    2022-05-11 21:09:12,844 INFO: -----------------SAVE:12epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 21:11:08,646 INFO: Epoch:[013/200]
    2022-05-11 21:11:08,646 INFO: Train Loss:0.885 | Acc:0.8477 | F1:0.1559
    2022-05-11 21:11:19,053 INFO: val Loss:0.746 | Acc:0.8480 | F1:0.1561
    2022-05-11 21:11:20,930 INFO: -----------------SAVE:13epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 21:13:17,184 INFO: Epoch:[014/200]
    2022-05-11 21:13:17,185 INFO: Train Loss:0.811 | Acc:0.8486 | F1:0.1582
    2022-05-11 21:13:27,575 INFO: val Loss:0.696 | Acc:0.8526 | F1:0.1754
    2022-05-11 21:13:29,568 INFO: -----------------SAVE:14epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 21:15:25,609 INFO: Epoch:[015/200]
    2022-05-11 21:15:25,609 INFO: Train Loss:0.741 | Acc:0.8492 | F1:0.1659
    2022-05-11 21:15:35,995 INFO: val Loss:0.643 | Acc:0.8550 | F1:0.1852
    2022-05-11 21:15:37,872 INFO: -----------------SAVE:15epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 21:17:33,785 INFO: Epoch:[016/200]
    2022-05-11 21:17:33,786 INFO: Train Loss:0.689 | Acc:0.8521 | F1:0.1884
    2022-05-11 21:17:44,165 INFO: val Loss:0.573 | Acc:0.8632 | F1:0.2228
    2022-05-11 21:17:46,117 INFO: -----------------SAVE:16epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 21:19:41,972 INFO: Epoch:[017/200]
    2022-05-11 21:19:41,972 INFO: Train Loss:0.638 | Acc:0.8577 | F1:0.2277
    2022-05-11 21:19:52,405 INFO: val Loss:0.540 | Acc:0.8772 | F1:0.2903
    2022-05-11 21:19:54,261 INFO: -----------------SAVE:17epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 21:21:49,867 INFO: Epoch:[018/200]
    2022-05-11 21:21:49,867 INFO: Train Loss:0.576 | Acc:0.8685 | F1:0.2829
    2022-05-11 21:22:00,299 INFO: val Loss:0.485 | Acc:0.8854 | F1:0.3347
    2022-05-11 21:22:02,190 INFO: -----------------SAVE:18epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 21:23:58,400 INFO: Epoch:[019/200]
    2022-05-11 21:23:58,400 INFO: Train Loss:0.521 | Acc:0.8787 | F1:0.3416
    2022-05-11 21:24:08,902 INFO: val Loss:0.446 | Acc:0.8936 | F1:0.3802
    2022-05-11 21:24:11,037 INFO: -----------------SAVE:19epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 21:26:09,995 INFO: Epoch:[020/200]
    2022-05-11 21:26:09,995 INFO: Train Loss:0.484 | Acc:0.8860 | F1:0.3945
    2022-05-11 21:26:21,323 INFO: val Loss:0.404 | Acc:0.9064 | F1:0.4670
    2022-05-11 21:26:23,266 INFO: -----------------SAVE:20epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:01<00:00,  1.76it/s]
    2022-05-11 21:28:24,910 INFO: Epoch:[021/200]
    2022-05-11 21:28:24,911 INFO: Train Loss:0.438 | Acc:0.8933 | F1:0.4331
    2022-05-11 21:28:35,520 INFO: val Loss:0.362 | Acc:0.9088 | F1:0.4806
    2022-05-11 21:28:37,480 INFO: -----------------SAVE:21epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 21:30:35,890 INFO: Epoch:[022/200]
    2022-05-11 21:30:35,891 INFO: Train Loss:0.419 | Acc:0.8983 | F1:0.4633
    2022-05-11 21:30:46,282 INFO: val Loss:0.328 | Acc:0.9123 | F1:0.4915
    2022-05-11 21:30:48,276 INFO: -----------------SAVE:22epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 21:32:44,819 INFO: Epoch:[023/200]
    2022-05-11 21:32:44,820 INFO: Train Loss:0.383 | Acc:0.9053 | F1:0.5046
    2022-05-11 21:32:55,252 INFO: val Loss:0.322 | Acc:0.9135 | F1:0.4961
    2022-05-11 21:32:57,222 INFO: -----------------SAVE:23epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 21:34:53,751 INFO: Epoch:[024/200]
    2022-05-11 21:34:53,751 INFO: Train Loss:0.360 | Acc:0.9100 | F1:0.5305
    2022-05-11 21:35:04,275 INFO: val Loss:0.310 | Acc:0.9181 | F1:0.5121
    2022-05-11 21:35:06,211 INFO: -----------------SAVE:24epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 21:37:02,425 INFO: Epoch:[025/200]
    2022-05-11 21:37:02,425 INFO: Train Loss:0.338 | Acc:0.9112 | F1:0.5506
    2022-05-11 21:37:12,774 INFO: val Loss:0.302 | Acc:0.9228 | F1:0.5728
    2022-05-11 21:37:14,858 INFO: -----------------SAVE:25epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 21:39:10,854 INFO: Epoch:[026/200]
    2022-05-11 21:39:10,854 INFO: Train Loss:0.318 | Acc:0.9191 | F1:0.6004
    2022-05-11 21:39:21,250 INFO: val Loss:0.265 | Acc:0.9287 | F1:0.5792
    2022-05-11 21:39:23,146 INFO: -----------------SAVE:26epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 21:41:18,876 INFO: Epoch:[027/200]
    2022-05-11 21:41:18,876 INFO: Train Loss:0.306 | Acc:0.9214 | F1:0.6120
    2022-05-11 21:41:29,294 INFO: val Loss:0.256 | Acc:0.9287 | F1:0.5841
    2022-05-11 21:41:31,214 INFO: -----------------SAVE:27epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 21:43:26,959 INFO: Epoch:[028/200]
    2022-05-11 21:43:26,959 INFO: Train Loss:0.308 | Acc:0.9185 | F1:0.6124
    2022-05-11 21:43:37,406 INFO: val Loss:0.238 | Acc:0.9310 | F1:0.6195
    2022-05-11 21:43:39,326 INFO: -----------------SAVE:28epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-11 21:45:36,041 INFO: Epoch:[029/200]
    2022-05-11 21:45:36,041 INFO: Train Loss:0.279 | Acc:0.9252 | F1:0.6344
    2022-05-11 21:45:47,670 INFO: val Loss:0.211 | Acc:0.9368 | F1:0.6629
    2022-05-11 21:45:49,621 INFO: -----------------SAVE:29epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-11 21:47:46,451 INFO: Epoch:[030/200]
    2022-05-11 21:47:46,452 INFO: Train Loss:0.263 | Acc:0.9299 | F1:0.6589
    2022-05-11 21:47:56,878 INFO: val Loss:0.287 | Acc:0.9240 | F1:0.6096
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 21:49:52,528 INFO: Epoch:[031/200]
    2022-05-11 21:49:52,528 INFO: Train Loss:0.263 | Acc:0.9307 | F1:0.6668
    2022-05-11 21:50:03,007 INFO: val Loss:0.198 | Acc:0.9404 | F1:0.6648
    2022-05-11 21:50:04,998 INFO: -----------------SAVE:31epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 21:52:00,849 INFO: Epoch:[032/200]
    2022-05-11 21:52:00,850 INFO: Train Loss:0.249 | Acc:0.9354 | F1:0.6975
    2022-05-11 21:52:11,299 INFO: val Loss:0.228 | Acc:0.9404 | F1:0.6805
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 21:54:07,048 INFO: Epoch:[033/200]
    2022-05-11 21:54:07,048 INFO: Train Loss:0.239 | Acc:0.9380 | F1:0.7022
    2022-05-11 21:54:17,423 INFO: val Loss:0.231 | Acc:0.9345 | F1:0.6702
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 21:56:13,380 INFO: Epoch:[034/200]
    2022-05-11 21:56:13,380 INFO: Train Loss:0.239 | Acc:0.9357 | F1:0.7219
    2022-05-11 21:56:23,766 INFO: val Loss:0.205 | Acc:0.9404 | F1:0.6544
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 21:58:19,562 INFO: Epoch:[035/200]
    2022-05-11 21:58:19,563 INFO: Train Loss:0.249 | Acc:0.9316 | F1:0.7014
    2022-05-11 21:58:30,281 INFO: val Loss:0.206 | Acc:0.9415 | F1:0.6967
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.84it/s]
    2022-05-11 22:00:26,285 INFO: Epoch:[036/200]
    2022-05-11 22:00:26,286 INFO: Train Loss:0.207 | Acc:0.9462 | F1:0.7586
    2022-05-11 22:00:36,665 INFO: val Loss:0.205 | Acc:0.9380 | F1:0.6784
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.77it/s]
    2022-05-11 22:02:37,376 INFO: Epoch:[037/200]
    2022-05-11 22:02:37,376 INFO: Train Loss:0.214 | Acc:0.9448 | F1:0.7578
    2022-05-11 22:02:49,252 INFO: val Loss:0.230 | Acc:0.9439 | F1:0.7171
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 22:04:45,868 INFO: Epoch:[038/200]
    2022-05-11 22:04:45,868 INFO: Train Loss:0.214 | Acc:0.9454 | F1:0.7710
    2022-05-11 22:04:56,227 INFO: val Loss:0.193 | Acc:0.9485 | F1:0.7302
    2022-05-11 22:04:58,019 INFO: -----------------SAVE:38epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 22:06:56,750 INFO: Epoch:[039/200]
    2022-05-11 22:06:56,750 INFO: Train Loss:0.223 | Acc:0.9418 | F1:0.7415
    2022-05-11 22:07:09,475 INFO: val Loss:0.167 | Acc:0.9497 | F1:0.7001
    2022-05-11 22:07:11,430 INFO: -----------------SAVE:39epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 22:09:10,189 INFO: Epoch:[040/200]
    2022-05-11 22:09:10,189 INFO: Train Loss:0.193 | Acc:0.9500 | F1:0.7800
    2022-05-11 22:09:20,686 INFO: val Loss:0.247 | Acc:0.9357 | F1:0.6933
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.80it/s]
    2022-05-11 22:11:19,813 INFO: Epoch:[041/200]
    2022-05-11 22:11:19,814 INFO: Train Loss:0.223 | Acc:0.9389 | F1:0.7403
    2022-05-11 22:11:30,640 INFO: val Loss:0.180 | Acc:0.9532 | F1:0.7283
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 22:13:26,972 INFO: Epoch:[042/200]
    2022-05-11 22:13:26,972 INFO: Train Loss:0.193 | Acc:0.9500 | F1:0.7846
    2022-05-11 22:13:37,401 INFO: val Loss:0.205 | Acc:0.9427 | F1:0.7049
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 22:15:33,540 INFO: Epoch:[043/200]
    2022-05-11 22:15:33,540 INFO: Train Loss:0.199 | Acc:0.9497 | F1:0.7868
    2022-05-11 22:15:43,938 INFO: val Loss:0.193 | Acc:0.9474 | F1:0.7023
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 22:17:40,164 INFO: Epoch:[044/200]
    2022-05-11 22:17:40,165 INFO: Train Loss:0.197 | Acc:0.9486 | F1:0.7867
    2022-05-11 22:17:50,554 INFO: val Loss:0.243 | Acc:0.9415 | F1:0.6636
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 22:19:46,815 INFO: Epoch:[045/200]
    2022-05-11 22:19:46,816 INFO: Train Loss:0.204 | Acc:0.9480 | F1:0.7793
    2022-05-11 22:19:57,384 INFO: val Loss:0.201 | Acc:0.9462 | F1:0.6951
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 22:21:53,722 INFO: Epoch:[046/200]
    2022-05-11 22:21:53,722 INFO: Train Loss:0.212 | Acc:0.9462 | F1:0.7699
    2022-05-11 22:22:04,301 INFO: val Loss:0.216 | Acc:0.9450 | F1:0.6941
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 22:24:00,444 INFO: Epoch:[047/200]
    2022-05-11 22:24:00,445 INFO: Train Loss:0.197 | Acc:0.9489 | F1:0.7950
    2022-05-11 22:24:10,962 INFO: val Loss:0.181 | Acc:0.9509 | F1:0.7396
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 22:26:07,222 INFO: Epoch:[048/200]
    2022-05-11 22:26:07,223 INFO: Train Loss:0.196 | Acc:0.9527 | F1:0.7997
    2022-05-11 22:26:17,650 INFO: val Loss:0.165 | Acc:0.9532 | F1:0.7384
    2022-05-11 22:26:19,607 INFO: -----------------SAVE:48epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 22:28:16,229 INFO: Epoch:[049/200]
    2022-05-11 22:28:16,229 INFO: Train Loss:0.179 | Acc:0.9521 | F1:0.7908
    2022-05-11 22:28:26,603 INFO: val Loss:0.206 | Acc:0.9450 | F1:0.6802
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 22:30:22,847 INFO: Epoch:[050/200]
    2022-05-11 22:30:22,848 INFO: Train Loss:0.202 | Acc:0.9451 | F1:0.7642
    2022-05-11 22:30:33,325 INFO: val Loss:0.239 | Acc:0.9462 | F1:0.6919
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.84it/s]
    2022-05-11 22:32:29,325 INFO: Epoch:[051/200]
    2022-05-11 22:32:29,325 INFO: Train Loss:0.179 | Acc:0.9524 | F1:0.8017
    2022-05-11 22:32:39,732 INFO: val Loss:0.189 | Acc:0.9532 | F1:0.7528
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 22:34:35,960 INFO: Epoch:[052/200]
    2022-05-11 22:34:35,960 INFO: Train Loss:0.184 | Acc:0.9509 | F1:0.8076
    2022-05-11 22:34:46,365 INFO: val Loss:0.201 | Acc:0.9497 | F1:0.7288
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 22:36:42,380 INFO: Epoch:[053/200]
    2022-05-11 22:36:42,380 INFO: Train Loss:0.179 | Acc:0.9527 | F1:0.7946
    2022-05-11 22:36:52,779 INFO: val Loss:0.189 | Acc:0.9567 | F1:0.7513
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 22:38:49,294 INFO: Epoch:[054/200]
    2022-05-11 22:38:49,294 INFO: Train Loss:0.186 | Acc:0.9497 | F1:0.8076
    2022-05-11 22:38:59,668 INFO: val Loss:0.200 | Acc:0.9520 | F1:0.7295
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 22:40:56,133 INFO: Epoch:[055/200]
    2022-05-11 22:40:56,133 INFO: Train Loss:0.183 | Acc:0.9521 | F1:0.8049
    2022-05-11 22:41:06,733 INFO: val Loss:0.179 | Acc:0.9591 | F1:0.7622
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 22:43:02,864 INFO: Epoch:[056/200]
    2022-05-11 22:43:02,864 INFO: Train Loss:0.175 | Acc:0.9506 | F1:0.8116
    2022-05-11 22:43:13,275 INFO: val Loss:0.212 | Acc:0.9509 | F1:0.7291
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 22:45:09,557 INFO: Epoch:[057/200]
    2022-05-11 22:45:09,557 INFO: Train Loss:0.160 | Acc:0.9541 | F1:0.8288
    2022-05-11 22:45:19,975 INFO: val Loss:0.221 | Acc:0.9532 | F1:0.7327
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 22:47:16,048 INFO: Epoch:[058/200]
    2022-05-11 22:47:16,048 INFO: Train Loss:0.165 | Acc:0.9530 | F1:0.8072
    2022-05-11 22:47:26,400 INFO: val Loss:0.233 | Acc:0.9497 | F1:0.7367
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 22:49:22,543 INFO: Epoch:[059/200]
    2022-05-11 22:49:22,544 INFO: Train Loss:0.175 | Acc:0.9532 | F1:0.8016
    2022-05-11 22:49:32,955 INFO: val Loss:0.167 | Acc:0.9637 | F1:0.8248
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 22:51:29,019 INFO: Epoch:[060/200]
    2022-05-11 22:51:29,020 INFO: Train Loss:0.167 | Acc:0.9556 | F1:0.8266
    2022-05-11 22:51:39,434 INFO: val Loss:0.201 | Acc:0.9462 | F1:0.7508
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 22:53:35,664 INFO: Epoch:[061/200]
    2022-05-11 22:53:35,664 INFO: Train Loss:0.168 | Acc:0.9594 | F1:0.8444
    2022-05-11 22:53:46,066 INFO: val Loss:0.302 | Acc:0.9345 | F1:0.7497
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 22:55:42,126 INFO: Epoch:[062/200]
    2022-05-11 22:55:42,127 INFO: Train Loss:0.167 | Acc:0.9565 | F1:0.8322
    2022-05-11 22:55:52,544 INFO: val Loss:0.334 | Acc:0.8982 | F1:0.7147
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 22:57:48,471 INFO: Epoch:[063/200]
    2022-05-11 22:57:48,471 INFO: Train Loss:0.163 | Acc:0.9570 | F1:0.8390
    2022-05-11 22:57:58,902 INFO: val Loss:0.351 | Acc:0.9123 | F1:0.7605
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.78it/s]
    2022-05-11 22:59:59,350 INFO: Epoch:[064/200]
    2022-05-11 22:59:59,350 INFO: Train Loss:0.162 | Acc:0.9565 | F1:0.8330
    2022-05-11 23:00:10,445 INFO: val Loss:0.288 | Acc:0.9275 | F1:0.7252
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.80it/s]
    2022-05-11 23:02:09,459 INFO: Epoch:[065/200]
    2022-05-11 23:02:09,459 INFO: Train Loss:0.163 | Acc:0.9582 | F1:0.8405
    2022-05-11 23:02:20,308 INFO: val Loss:0.143 | Acc:0.9626 | F1:0.8052
    2022-05-11 23:02:22,167 INFO: -----------------SAVE:65epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 23:04:20,907 INFO: Epoch:[066/200]
    2022-05-11 23:04:20,907 INFO: Train Loss:0.154 | Acc:0.9597 | F1:0.8447
    2022-05-11 23:04:31,293 INFO: val Loss:0.165 | Acc:0.9532 | F1:0.7684
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 23:06:27,232 INFO: Epoch:[067/200]
    2022-05-11 23:06:27,232 INFO: Train Loss:0.139 | Acc:0.9626 | F1:0.8474
    2022-05-11 23:06:37,667 INFO: val Loss:0.175 | Acc:0.9556 | F1:0.8084
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 23:08:33,634 INFO: Epoch:[068/200]
    2022-05-11 23:08:33,634 INFO: Train Loss:0.138 | Acc:0.9635 | F1:0.8614
    2022-05-11 23:08:44,041 INFO: val Loss:0.161 | Acc:0.9614 | F1:0.8245
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-11 23:10:41,766 INFO: Epoch:[069/200]
    2022-05-11 23:10:41,767 INFO: Train Loss:0.132 | Acc:0.9652 | F1:0.8690
    2022-05-11 23:10:52,402 INFO: val Loss:0.150 | Acc:0.9637 | F1:0.8016
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.80it/s]
    2022-05-11 23:12:51,496 INFO: Epoch:[070/200]
    2022-05-11 23:12:51,497 INFO: Train Loss:0.124 | Acc:0.9679 | F1:0.8742
    2022-05-11 23:13:02,297 INFO: val Loss:0.166 | Acc:0.9579 | F1:0.7620
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 23:15:01,132 INFO: Epoch:[071/200]
    2022-05-11 23:15:01,132 INFO: Train Loss:0.131 | Acc:0.9635 | F1:0.8662
    2022-05-11 23:15:12,012 INFO: val Loss:0.220 | Acc:0.9485 | F1:0.7544
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-11 23:17:08,780 INFO: Epoch:[072/200]
    2022-05-11 23:17:08,781 INFO: Train Loss:0.148 | Acc:0.9632 | F1:0.8487
    2022-05-11 23:17:19,269 INFO: val Loss:0.183 | Acc:0.9602 | F1:0.8004
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 23:19:15,751 INFO: Epoch:[073/200]
    2022-05-11 23:19:15,751 INFO: Train Loss:0.146 | Acc:0.9591 | F1:0.8440
    2022-05-11 23:19:26,231 INFO: val Loss:0.206 | Acc:0.9520 | F1:0.7578
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.83it/s]
    2022-05-11 23:21:23,473 INFO: Epoch:[074/200]
    2022-05-11 23:21:23,474 INFO: Train Loss:0.144 | Acc:0.9649 | F1:0.8721
    2022-05-11 23:21:33,937 INFO: val Loss:0.252 | Acc:0.9404 | F1:0.7043
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-11 23:23:30,576 INFO: Epoch:[075/200]
    2022-05-11 23:23:30,577 INFO: Train Loss:0.139 | Acc:0.9620 | F1:0.8449
    2022-05-11 23:23:41,037 INFO: val Loss:0.196 | Acc:0.9591 | F1:0.7832
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 23:25:37,285 INFO: Epoch:[076/200]
    2022-05-11 23:25:37,285 INFO: Train Loss:0.133 | Acc:0.9643 | F1:0.8604
    2022-05-11 23:25:48,378 INFO: val Loss:0.198 | Acc:0.9520 | F1:0.7564
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-11 23:27:45,268 INFO: Epoch:[077/200]
    2022-05-11 23:27:45,268 INFO: Train Loss:0.139 | Acc:0.9649 | F1:0.8685
    2022-05-11 23:27:55,799 INFO: val Loss:0.165 | Acc:0.9661 | F1:0.8446
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-11 23:29:54,370 INFO: Epoch:[078/200]
    2022-05-11 23:29:54,370 INFO: Train Loss:0.135 | Acc:0.9670 | F1:0.8722
    2022-05-11 23:30:04,928 INFO: val Loss:0.197 | Acc:0.9532 | F1:0.7863
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 23:32:01,417 INFO: Epoch:[079/200]
    2022-05-11 23:32:01,418 INFO: Train Loss:0.142 | Acc:0.9673 | F1:0.8679
    2022-05-11 23:32:11,858 INFO: val Loss:0.181 | Acc:0.9556 | F1:0.7711
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.81it/s]
    2022-05-11 23:34:09,809 INFO: Epoch:[080/200]
    2022-05-11 23:34:09,809 INFO: Train Loss:0.130 | Acc:0.9676 | F1:0.8813
    2022-05-11 23:34:20,890 INFO: val Loss:0.171 | Acc:0.9556 | F1:0.7478
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-11 23:36:19,240 INFO: Epoch:[081/200]
    2022-05-11 23:36:19,240 INFO: Train Loss:0.131 | Acc:0.9667 | F1:0.8702
    2022-05-11 23:36:29,717 INFO: val Loss:0.160 | Acc:0.9556 | F1:0.7808
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-11 23:38:26,359 INFO: Epoch:[082/200]
    2022-05-11 23:38:26,359 INFO: Train Loss:0.122 | Acc:0.9684 | F1:0.8823
    2022-05-11 23:38:36,814 INFO: val Loss:0.147 | Acc:0.9579 | F1:0.7729
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 23:40:33,352 INFO: Epoch:[083/200]
    2022-05-11 23:40:33,352 INFO: Train Loss:0.121 | Acc:0.9667 | F1:0.8720
    2022-05-11 23:40:43,864 INFO: val Loss:0.150 | Acc:0.9602 | F1:0.7861
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-11 23:42:43,732 INFO: Epoch:[084/200]
    2022-05-11 23:42:43,733 INFO: Train Loss:0.115 | Acc:0.9693 | F1:0.8856
    2022-05-11 23:42:55,071 INFO: val Loss:0.154 | Acc:0.9556 | F1:0.7700
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 23:44:51,395 INFO: Epoch:[085/200]
    2022-05-11 23:44:51,395 INFO: Train Loss:0.117 | Acc:0.9714 | F1:0.9040
    2022-05-11 23:45:01,907 INFO: val Loss:0.169 | Acc:0.9602 | F1:0.7609
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-11 23:46:59,279 INFO: Epoch:[086/200]
    2022-05-11 23:46:59,280 INFO: Train Loss:0.114 | Acc:0.9679 | F1:0.8737
    2022-05-11 23:47:09,802 INFO: val Loss:0.167 | Acc:0.9579 | F1:0.7708
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 23:49:06,137 INFO: Epoch:[087/200]
    2022-05-11 23:49:06,137 INFO: Train Loss:0.126 | Acc:0.9687 | F1:0.8773
    2022-05-11 23:49:16,611 INFO: val Loss:0.209 | Acc:0.9520 | F1:0.7599
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 23:51:12,663 INFO: Epoch:[088/200]
    2022-05-11 23:51:12,664 INFO: Train Loss:0.111 | Acc:0.9719 | F1:0.8955
    2022-05-11 23:51:23,120 INFO: val Loss:0.150 | Acc:0.9591 | F1:0.7802
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-11 23:53:19,076 INFO: Epoch:[089/200]
    2022-05-11 23:53:19,076 INFO: Train Loss:0.122 | Acc:0.9690 | F1:0.8824
    2022-05-11 23:53:29,579 INFO: val Loss:0.171 | Acc:0.9591 | F1:0.7902
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 23:55:25,905 INFO: Epoch:[090/200]
    2022-05-11 23:55:25,905 INFO: Train Loss:0.117 | Acc:0.9681 | F1:0.8742
    2022-05-11 23:55:36,474 INFO: val Loss:0.165 | Acc:0.9579 | F1:0.7890
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-11 23:57:33,147 INFO: Epoch:[091/200]
    2022-05-11 23:57:33,148 INFO: Train Loss:0.119 | Acc:0.9661 | F1:0.8761
    2022-05-11 23:57:43,658 INFO: val Loss:0.132 | Acc:0.9696 | F1:0.8196
    2022-05-11 23:57:45,927 INFO: -----------------SAVE:91epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-11 23:59:42,354 INFO: Epoch:[092/200]
    2022-05-11 23:59:42,355 INFO: Train Loss:0.110 | Acc:0.9708 | F1:0.8976
    2022-05-11 23:59:52,910 INFO: val Loss:0.171 | Acc:0.9614 | F1:0.7801
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 00:01:49,679 INFO: Epoch:[093/200]
    2022-05-12 00:01:49,679 INFO: Train Loss:0.110 | Acc:0.9725 | F1:0.8969
    2022-05-12 00:02:00,184 INFO: val Loss:0.147 | Acc:0.9626 | F1:0.8097
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.83it/s]
    2022-05-12 00:03:57,385 INFO: Epoch:[094/200]
    2022-05-12 00:03:57,385 INFO: Train Loss:0.110 | Acc:0.9725 | F1:0.9025
    2022-05-12 00:04:08,151 INFO: val Loss:0.149 | Acc:0.9626 | F1:0.8129
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 00:06:05,078 INFO: Epoch:[095/200]
    2022-05-12 00:06:05,079 INFO: Train Loss:0.091 | Acc:0.9740 | F1:0.9012
    2022-05-12 00:06:15,493 INFO: val Loss:0.179 | Acc:0.9567 | F1:0.8151
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 00:08:11,809 INFO: Epoch:[096/200]
    2022-05-12 00:08:11,810 INFO: Train Loss:0.092 | Acc:0.9763 | F1:0.9063
    2022-05-12 00:08:22,204 INFO: val Loss:0.214 | Acc:0.9544 | F1:0.7717
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 00:10:18,275 INFO: Epoch:[097/200]
    2022-05-12 00:10:18,276 INFO: Train Loss:0.114 | Acc:0.9719 | F1:0.9003
    2022-05-12 00:10:28,776 INFO: val Loss:0.199 | Acc:0.9567 | F1:0.8018
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 00:12:24,835 INFO: Epoch:[098/200]
    2022-05-12 00:12:24,836 INFO: Train Loss:0.094 | Acc:0.9737 | F1:0.9070
    2022-05-12 00:12:35,334 INFO: val Loss:0.177 | Acc:0.9567 | F1:0.7794
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 00:14:31,510 INFO: Epoch:[099/200]
    2022-05-12 00:14:31,510 INFO: Train Loss:0.108 | Acc:0.9714 | F1:0.8951
    2022-05-12 00:14:42,014 INFO: val Loss:0.157 | Acc:0.9614 | F1:0.7971
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-12 00:16:40,113 INFO: Epoch:[100/200]
    2022-05-12 00:16:40,113 INFO: Train Loss:0.104 | Acc:0.9722 | F1:0.8950
    2022-05-12 00:16:50,567 INFO: val Loss:0.152 | Acc:0.9591 | F1:0.7909
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 00:18:47,328 INFO: Epoch:[101/200]
    2022-05-12 00:18:47,328 INFO: Train Loss:0.097 | Acc:0.9728 | F1:0.8978
    2022-05-12 00:18:57,748 INFO: val Loss:0.182 | Acc:0.9637 | F1:0.7945
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 00:20:53,903 INFO: Epoch:[102/200]
    2022-05-12 00:20:53,904 INFO: Train Loss:0.088 | Acc:0.9757 | F1:0.9084
    2022-05-12 00:21:04,422 INFO: val Loss:0.183 | Acc:0.9602 | F1:0.7851
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 00:23:00,582 INFO: Epoch:[103/200]
    2022-05-12 00:23:00,583 INFO: Train Loss:0.088 | Acc:0.9781 | F1:0.9219
    2022-05-12 00:23:11,150 INFO: val Loss:0.200 | Acc:0.9602 | F1:0.7954
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 00:25:07,511 INFO: Epoch:[104/200]
    2022-05-12 00:25:07,512 INFO: Train Loss:0.095 | Acc:0.9722 | F1:0.8994
    2022-05-12 00:25:17,958 INFO: val Loss:0.149 | Acc:0.9708 | F1:0.8666
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 00:27:14,123 INFO: Epoch:[105/200]
    2022-05-12 00:27:14,124 INFO: Train Loss:0.086 | Acc:0.9787 | F1:0.9232
    2022-05-12 00:27:24,783 INFO: val Loss:0.147 | Acc:0.9684 | F1:0.8455
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 00:29:21,373 INFO: Epoch:[106/200]
    2022-05-12 00:29:21,373 INFO: Train Loss:0.107 | Acc:0.9725 | F1:0.8981
    2022-05-12 00:29:31,875 INFO: val Loss:0.160 | Acc:0.9626 | F1:0.8388
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 00:31:28,150 INFO: Epoch:[107/200]
    2022-05-12 00:31:28,150 INFO: Train Loss:0.093 | Acc:0.9728 | F1:0.9064
    2022-05-12 00:31:38,643 INFO: val Loss:0.214 | Acc:0.9591 | F1:0.8398
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 00:33:35,204 INFO: Epoch:[108/200]
    2022-05-12 00:33:35,205 INFO: Train Loss:0.084 | Acc:0.9793 | F1:0.9341
    2022-05-12 00:33:45,713 INFO: val Loss:0.158 | Acc:0.9661 | F1:0.8286
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 00:35:42,423 INFO: Epoch:[109/200]
    2022-05-12 00:35:42,423 INFO: Train Loss:0.078 | Acc:0.9781 | F1:0.9326
    2022-05-12 00:35:52,899 INFO: val Loss:0.149 | Acc:0.9637 | F1:0.8343
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 00:37:49,088 INFO: Epoch:[110/200]
    2022-05-12 00:37:49,088 INFO: Train Loss:0.091 | Acc:0.9772 | F1:0.9194
    2022-05-12 00:37:59,568 INFO: val Loss:0.143 | Acc:0.9696 | F1:0.8289
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 00:39:55,956 INFO: Epoch:[111/200]
    2022-05-12 00:39:55,957 INFO: Train Loss:0.086 | Acc:0.9775 | F1:0.9216
    2022-05-12 00:40:06,576 INFO: val Loss:0.137 | Acc:0.9637 | F1:0.8044
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 00:42:03,411 INFO: Epoch:[112/200]
    2022-05-12 00:42:03,411 INFO: Train Loss:0.073 | Acc:0.9795 | F1:0.9295
    2022-05-12 00:42:13,869 INFO: val Loss:0.140 | Acc:0.9708 | F1:0.8425
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 00:44:10,416 INFO: Epoch:[113/200]
    2022-05-12 00:44:10,416 INFO: Train Loss:0.088 | Acc:0.9772 | F1:0.9185
    2022-05-12 00:44:20,843 INFO: val Loss:0.170 | Acc:0.9673 | F1:0.8285
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 00:46:17,263 INFO: Epoch:[114/200]
    2022-05-12 00:46:17,264 INFO: Train Loss:0.082 | Acc:0.9784 | F1:0.9248
    2022-05-12 00:46:27,779 INFO: val Loss:0.255 | Acc:0.9427 | F1:0.8351
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 00:48:24,335 INFO: Epoch:[115/200]
    2022-05-12 00:48:24,336 INFO: Train Loss:0.075 | Acc:0.9790 | F1:0.9209
    2022-05-12 00:48:34,785 INFO: val Loss:0.169 | Acc:0.9556 | F1:0.7802
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-12 00:50:34,158 INFO: Epoch:[116/200]
    2022-05-12 00:50:34,159 INFO: Train Loss:0.076 | Acc:0.9763 | F1:0.9075
    2022-05-12 00:50:44,880 INFO: val Loss:0.143 | Acc:0.9673 | F1:0.8373
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 00:52:42,577 INFO: Epoch:[117/200]
    2022-05-12 00:52:42,577 INFO: Train Loss:0.071 | Acc:0.9790 | F1:0.9233
    2022-05-12 00:52:53,024 INFO: val Loss:0.137 | Acc:0.9661 | F1:0.8055
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:55<00:00,  1.85it/s]
    2022-05-12 00:54:48,993 INFO: Epoch:[118/200]
    2022-05-12 00:54:48,993 INFO: Train Loss:0.077 | Acc:0.9784 | F1:0.9265
    2022-05-12 00:54:59,409 INFO: val Loss:0.164 | Acc:0.9637 | F1:0.8173
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 00:56:55,746 INFO: Epoch:[119/200]
    2022-05-12 00:56:55,746 INFO: Train Loss:0.072 | Acc:0.9833 | F1:0.9402
    2022-05-12 00:57:06,276 INFO: val Loss:0.197 | Acc:0.9661 | F1:0.8270
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 00:59:02,499 INFO: Epoch:[120/200]
    2022-05-12 00:59:02,500 INFO: Train Loss:0.077 | Acc:0.9787 | F1:0.9202
    2022-05-12 00:59:12,984 INFO: val Loss:0.169 | Acc:0.9614 | F1:0.7948
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 01:01:09,601 INFO: Epoch:[121/200]
    2022-05-12 01:01:09,601 INFO: Train Loss:0.065 | Acc:0.9825 | F1:0.9366
    2022-05-12 01:01:20,070 INFO: val Loss:0.176 | Acc:0.9661 | F1:0.8152
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-12 01:03:18,123 INFO: Epoch:[122/200]
    2022-05-12 01:03:18,124 INFO: Train Loss:0.063 | Acc:0.9828 | F1:0.9406
    2022-05-12 01:03:28,574 INFO: val Loss:0.178 | Acc:0.9661 | F1:0.8280
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 01:05:25,126 INFO: Epoch:[123/200]
    2022-05-12 01:05:25,126 INFO: Train Loss:0.066 | Acc:0.9795 | F1:0.9260
    2022-05-12 01:05:35,588 INFO: val Loss:0.172 | Acc:0.9602 | F1:0.8207
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 01:07:31,952 INFO: Epoch:[124/200]
    2022-05-12 01:07:31,953 INFO: Train Loss:0.056 | Acc:0.9828 | F1:0.9446
    2022-05-12 01:07:42,433 INFO: val Loss:0.177 | Acc:0.9614 | F1:0.8128
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 01:09:38,866 INFO: Epoch:[125/200]
    2022-05-12 01:09:38,866 INFO: Train Loss:0.066 | Acc:0.9816 | F1:0.9292
    2022-05-12 01:09:49,353 INFO: val Loss:0.156 | Acc:0.9626 | F1:0.7940
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 01:11:45,818 INFO: Epoch:[126/200]
    2022-05-12 01:11:45,819 INFO: Train Loss:0.066 | Acc:0.9807 | F1:0.9357
    2022-05-12 01:11:56,243 INFO: val Loss:0.147 | Acc:0.9661 | F1:0.8313
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-12 01:13:55,733 INFO: Epoch:[127/200]
    2022-05-12 01:13:55,733 INFO: Train Loss:0.073 | Acc:0.9825 | F1:0.9384
    2022-05-12 01:14:06,853 INFO: val Loss:0.147 | Acc:0.9649 | F1:0.8373
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.83it/s]
    2022-05-12 01:16:03,984 INFO: Epoch:[128/200]
    2022-05-12 01:16:03,985 INFO: Train Loss:0.061 | Acc:0.9822 | F1:0.9395
    2022-05-12 01:16:14,458 INFO: val Loss:0.142 | Acc:0.9673 | F1:0.8219
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.81it/s]
    2022-05-12 01:18:12,459 INFO: Epoch:[129/200]
    2022-05-12 01:18:12,459 INFO: Train Loss:0.063 | Acc:0.9833 | F1:0.9441
    2022-05-12 01:18:22,947 INFO: val Loss:0.160 | Acc:0.9684 | F1:0.8582
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.80it/s]
    2022-05-12 01:20:22,004 INFO: Epoch:[130/200]
    2022-05-12 01:20:22,005 INFO: Train Loss:0.069 | Acc:0.9822 | F1:0.9388
    2022-05-12 01:20:32,683 INFO: val Loss:0.175 | Acc:0.9614 | F1:0.8281
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.78it/s]
    2022-05-12 01:22:32,787 INFO: Epoch:[131/200]
    2022-05-12 01:22:32,787 INFO: Train Loss:0.062 | Acc:0.9831 | F1:0.9430
    2022-05-12 01:22:43,999 INFO: val Loss:0.142 | Acc:0.9684 | F1:0.8384
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.83it/s]
    2022-05-12 01:24:41,163 INFO: Epoch:[132/200]
    2022-05-12 01:24:41,163 INFO: Train Loss:0.046 | Acc:0.9883 | F1:0.9648
    2022-05-12 01:24:51,616 INFO: val Loss:0.139 | Acc:0.9684 | F1:0.8462
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 01:26:48,052 INFO: Epoch:[133/200]
    2022-05-12 01:26:48,052 INFO: Train Loss:0.060 | Acc:0.9836 | F1:0.9404
    2022-05-12 01:26:58,524 INFO: val Loss:0.152 | Acc:0.9637 | F1:0.8413
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 01:28:55,241 INFO: Epoch:[134/200]
    2022-05-12 01:28:55,241 INFO: Train Loss:0.054 | Acc:0.9836 | F1:0.9466
    2022-05-12 01:29:05,772 INFO: val Loss:0.169 | Acc:0.9661 | F1:0.8179
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 01:31:02,165 INFO: Epoch:[135/200]
    2022-05-12 01:31:02,166 INFO: Train Loss:0.041 | Acc:0.9895 | F1:0.9645
    2022-05-12 01:31:12,596 INFO: val Loss:0.142 | Acc:0.9719 | F1:0.8536
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.80it/s]
    2022-05-12 01:33:11,724 INFO: Epoch:[136/200]
    2022-05-12 01:33:11,725 INFO: Train Loss:0.067 | Acc:0.9831 | F1:0.9453
    2022-05-12 01:33:22,652 INFO: val Loss:0.160 | Acc:0.9649 | F1:0.8034
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-12 01:35:21,005 INFO: Epoch:[137/200]
    2022-05-12 01:35:21,005 INFO: Train Loss:0.058 | Acc:0.9836 | F1:0.9369
    2022-05-12 01:35:31,561 INFO: val Loss:0.163 | Acc:0.9649 | F1:0.8186
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 01:37:28,351 INFO: Epoch:[138/200]
    2022-05-12 01:37:28,351 INFO: Train Loss:0.052 | Acc:0.9857 | F1:0.9532
    2022-05-12 01:37:38,804 INFO: val Loss:0.171 | Acc:0.9637 | F1:0.7949
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 01:39:35,241 INFO: Epoch:[139/200]
    2022-05-12 01:39:35,243 INFO: Train Loss:0.050 | Acc:0.9860 | F1:0.9519
    2022-05-12 01:39:45,689 INFO: val Loss:0.153 | Acc:0.9719 | F1:0.8277
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 01:41:42,086 INFO: Epoch:[140/200]
    2022-05-12 01:41:42,086 INFO: Train Loss:0.072 | Acc:0.9822 | F1:0.9373
    2022-05-12 01:41:52,504 INFO: val Loss:0.146 | Acc:0.9661 | F1:0.8127
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 01:43:48,865 INFO: Epoch:[141/200]
    2022-05-12 01:43:48,866 INFO: Train Loss:0.053 | Acc:0.9863 | F1:0.9534
    2022-05-12 01:43:59,391 INFO: val Loss:0.179 | Acc:0.9579 | F1:0.7824
    2022-05-12 01:43:59,391 INFO: 
    Best Val Epoch:91 | Val Loss:0.1316 | Val Acc:0.9696 | Val F1:0.8196
    2022-05-12 01:43:59,392 INFO: Total Process time:300.714Minute
    2022-05-12 01:43:59,400 INFO: {'exp_num': '4', 'data_path': './data', 'Kfold': 5, 'model_path': 'results/', 'encoder_name': 'regnety_160', 'drop_path_rate': 0.2, 'img_size': 224, 'batch_size': 16, 'epochs': 200, 'optimizer': 'Lamb', 'initial_lr': 5e-06, 'weight_decay': 0.001, 'aug_ver': 2, 'scheduler': 'cycle', 'warm_epoch': 5, 'max_lr': 0.001, 'min_lr': 5e-06, 'tmax': 145, 'patience': 50, 'clipping': None, 'amp': True, 'multi_gpu': False, 'logging': False, 'num_workers': 0, 'seed': 42, 'fold': 4}
    

    <---- Training Params ---->
    Read train_df.csv
    Dataset size:3422
    Dataset size:855
    

    2022-05-12 01:44:00,260 INFO: Loading pretrained weights from url (https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth)
    2022-05-12 01:44:00,657 INFO: Computational complexity:       15.93 GMac
    2022-05-12 01:44:00,658 INFO: Number of parameters:           80.83 M 
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 01:45:57,058 INFO: Epoch:[001/200]
    2022-05-12 01:45:57,058 INFO: Train Loss:4.487 | Acc:0.0117 | F1:0.0053
    2022-05-12 01:46:07,657 INFO: val Loss:4.542 | Acc:0.0000 | F1:0.0000
    2022-05-12 01:46:09,519 INFO: -----------------SAVE:1epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 01:48:06,840 INFO: Epoch:[002/200]
    2022-05-12 01:48:06,840 INFO: Train Loss:4.469 | Acc:0.0129 | F1:0.0058
    2022-05-12 01:48:17,668 INFO: val Loss:4.526 | Acc:0.0012 | F1:0.0002
    2022-05-12 01:48:19,631 INFO: -----------------SAVE:2epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:02<00:00,  1.74it/s]
    2022-05-12 01:50:22,522 INFO: Epoch:[003/200]
    2022-05-12 01:50:22,522 INFO: Train Loss:4.444 | Acc:0.0216 | F1:0.0077
    2022-05-12 01:50:34,083 INFO: val Loss:4.500 | Acc:0.0023 | F1:0.0003
    2022-05-12 01:50:35,911 INFO: -----------------SAVE:3epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:02<00:00,  1.75it/s]
    2022-05-12 01:52:38,492 INFO: Epoch:[004/200]
    2022-05-12 01:52:38,493 INFO: Train Loss:4.402 | Acc:0.0383 | F1:0.0115
    2022-05-12 01:52:49,864 INFO: val Loss:4.465 | Acc:0.0105 | F1:0.0016
    2022-05-12 01:52:51,687 INFO: -----------------SAVE:4epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:01<00:00,  1.75it/s]
    2022-05-12 01:54:53,685 INFO: Epoch:[005/200]
    2022-05-12 01:54:53,686 INFO: Train Loss:4.346 | Acc:0.0663 | F1:0.0160
    2022-05-12 01:55:05,369 INFO: val Loss:4.395 | Acc:0.0234 | F1:0.0037
    2022-05-12 01:55:07,322 INFO: -----------------SAVE:5epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.78it/s]
    2022-05-12 01:57:07,800 INFO: Epoch:[006/200]
    2022-05-12 01:57:07,800 INFO: Train Loss:4.019 | Acc:0.3676 | F1:0.0747
    2022-05-12 01:57:18,819 INFO: val Loss:3.791 | Acc:0.5942 | F1:0.1175
    2022-05-12 01:57:20,745 INFO: -----------------SAVE:6epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:01<00:00,  1.76it/s]
    2022-05-12 01:59:22,613 INFO: Epoch:[007/200]
    2022-05-12 01:59:22,613 INFO: Train Loss:3.236 | Acc:0.7554 | F1:0.1425
    2022-05-12 01:59:33,734 INFO: val Loss:2.746 | Acc:0.7731 | F1:0.1396
    2022-05-12 01:59:35,677 INFO: -----------------SAVE:7epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.77it/s]
    2022-05-12 02:01:36,513 INFO: Epoch:[008/200]
    2022-05-12 02:01:36,513 INFO: Train Loss:2.342 | Acc:0.8250 | F1:0.1456
    2022-05-12 02:01:47,559 INFO: val Loss:1.719 | Acc:0.8374 | F1:0.1510
    2022-05-12 02:01:49,569 INFO: -----------------SAVE:8epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.77it/s]
    2022-05-12 02:03:50,385 INFO: Epoch:[009/200]
    2022-05-12 02:03:50,385 INFO: Train Loss:1.618 | Acc:0.8331 | F1:0.1505
    2022-05-12 02:04:01,613 INFO: val Loss:1.100 | Acc:0.8398 | F1:0.1528
    2022-05-12 02:04:03,822 INFO: -----------------SAVE:9epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:01<00:00,  1.76it/s]
    2022-05-12 02:06:05,542 INFO: Epoch:[010/200]
    2022-05-12 02:06:05,543 INFO: Train Loss:1.270 | Acc:0.8378 | F1:0.1516
    2022-05-12 02:06:16,846 INFO: val Loss:1.018 | Acc:0.8480 | F1:0.1578
    2022-05-12 02:06:18,855 INFO: -----------------SAVE:10epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.77it/s]
    2022-05-12 02:08:19,552 INFO: Epoch:[011/200]
    2022-05-12 02:08:19,552 INFO: Train Loss:1.091 | Acc:0.8457 | F1:0.1554
    2022-05-12 02:08:30,692 INFO: val Loss:0.914 | Acc:0.8480 | F1:0.1578
    2022-05-12 02:08:32,592 INFO: -----------------SAVE:11epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.78it/s]
    2022-05-12 02:10:32,964 INFO: Epoch:[012/200]
    2022-05-12 02:10:32,964 INFO: Train Loss:0.985 | Acc:0.8477 | F1:0.1560
    2022-05-12 02:10:44,043 INFO: val Loss:0.833 | Acc:0.8480 | F1:0.1578
    2022-05-12 02:10:45,985 INFO: -----------------SAVE:12epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:01<00:00,  1.76it/s]
    2022-05-12 02:12:47,691 INFO: Epoch:[013/200]
    2022-05-12 02:12:47,691 INFO: Train Loss:0.880 | Acc:0.8475 | F1:0.1560
    2022-05-12 02:12:58,956 INFO: val Loss:0.730 | Acc:0.8480 | F1:0.1578
    2022-05-12 02:13:01,259 INFO: -----------------SAVE:13epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:02<00:00,  1.75it/s]
    2022-05-12 02:15:03,567 INFO: Epoch:[014/200]
    2022-05-12 02:15:03,567 INFO: Train Loss:0.807 | Acc:0.8475 | F1:0.1558
    2022-05-12 02:15:14,922 INFO: val Loss:0.696 | Acc:0.8515 | F1:0.1696
    2022-05-12 02:15:17,585 INFO: -----------------SAVE:14epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:02<00:00,  1.75it/s]
    2022-05-12 02:17:19,817 INFO: Epoch:[015/200]
    2022-05-12 02:17:19,817 INFO: Train Loss:0.731 | Acc:0.8489 | F1:0.1653
    2022-05-12 02:17:31,176 INFO: val Loss:0.639 | Acc:0.8538 | F1:0.1929
    2022-05-12 02:17:33,046 INFO: -----------------SAVE:15epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:02<00:00,  1.75it/s]
    2022-05-12 02:19:35,526 INFO: Epoch:[016/200]
    2022-05-12 02:19:35,527 INFO: Train Loss:0.683 | Acc:0.8536 | F1:0.1916
    2022-05-12 02:19:46,802 INFO: val Loss:0.585 | Acc:0.8561 | F1:0.1932
    2022-05-12 02:19:48,656 INFO: -----------------SAVE:16epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:02<00:00,  1.75it/s]
    2022-05-12 02:21:51,194 INFO: Epoch:[017/200]
    2022-05-12 02:21:51,195 INFO: Train Loss:0.628 | Acc:0.8589 | F1:0.2231
    2022-05-12 02:22:02,437 INFO: val Loss:0.541 | Acc:0.8713 | F1:0.2807
    2022-05-12 02:22:04,320 INFO: -----------------SAVE:17epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:02<00:00,  1.75it/s]
    2022-05-12 02:24:06,652 INFO: Epoch:[018/200]
    2022-05-12 02:24:06,652 INFO: Train Loss:0.580 | Acc:0.8679 | F1:0.2736
    2022-05-12 02:24:18,001 INFO: val Loss:0.494 | Acc:0.8830 | F1:0.3528
    2022-05-12 02:24:19,789 INFO: -----------------SAVE:18epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:02<00:00,  1.75it/s]
    2022-05-12 02:26:22,156 INFO: Epoch:[019/200]
    2022-05-12 02:26:22,156 INFO: Train Loss:0.530 | Acc:0.8776 | F1:0.3358
    2022-05-12 02:26:33,554 INFO: val Loss:0.439 | Acc:0.8912 | F1:0.4048
    2022-05-12 02:26:35,539 INFO: -----------------SAVE:19epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:02<00:00,  1.74it/s]
    2022-05-12 02:28:38,240 INFO: Epoch:[020/200]
    2022-05-12 02:28:38,240 INFO: Train Loss:0.491 | Acc:0.8857 | F1:0.3820
    2022-05-12 02:28:49,548 INFO: val Loss:0.396 | Acc:0.8982 | F1:0.4259
    2022-05-12 02:28:51,849 INFO: -----------------SAVE:20epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-12 02:30:51,287 INFO: Epoch:[021/200]
    2022-05-12 02:30:51,288 INFO: Train Loss:0.445 | Acc:0.8939 | F1:0.4319
    2022-05-12 02:31:01,815 INFO: val Loss:0.357 | Acc:0.9146 | F1:0.5076
    2022-05-12 02:31:03,768 INFO: -----------------SAVE:21epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 02:33:00,474 INFO: Epoch:[022/200]
    2022-05-12 02:33:00,475 INFO: Train Loss:0.407 | Acc:0.8989 | F1:0.4659
    2022-05-12 02:33:10,917 INFO: val Loss:0.315 | Acc:0.9170 | F1:0.5281
    2022-05-12 02:33:12,893 INFO: -----------------SAVE:22epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 02:35:09,764 INFO: Epoch:[023/200]
    2022-05-12 02:35:09,765 INFO: Train Loss:0.375 | Acc:0.9053 | F1:0.5092
    2022-05-12 02:35:20,248 INFO: val Loss:0.321 | Acc:0.9146 | F1:0.5079
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 02:37:16,730 INFO: Epoch:[024/200]
    2022-05-12 02:37:16,730 INFO: Train Loss:0.346 | Acc:0.9117 | F1:0.5558
    2022-05-12 02:37:27,156 INFO: val Loss:0.283 | Acc:0.9275 | F1:0.5954
    2022-05-12 02:37:29,033 INFO: -----------------SAVE:24epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 02:39:25,921 INFO: Epoch:[025/200]
    2022-05-12 02:39:25,922 INFO: Train Loss:0.338 | Acc:0.9135 | F1:0.5586
    2022-05-12 02:39:36,368 INFO: val Loss:0.270 | Acc:0.9240 | F1:0.5874
    2022-05-12 02:39:38,394 INFO: -----------------SAVE:25epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 02:41:34,832 INFO: Epoch:[026/200]
    2022-05-12 02:41:34,832 INFO: Train Loss:0.307 | Acc:0.9193 | F1:0.5999
    2022-05-12 02:41:45,340 INFO: val Loss:0.258 | Acc:0.9310 | F1:0.6060
    2022-05-12 02:41:47,222 INFO: -----------------SAVE:26epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-12 02:43:46,210 INFO: Epoch:[027/200]
    2022-05-12 02:43:46,210 INFO: Train Loss:0.283 | Acc:0.9261 | F1:0.6284
    2022-05-12 02:43:56,653 INFO: val Loss:0.265 | Acc:0.9357 | F1:0.6384
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 02:45:53,135 INFO: Epoch:[028/200]
    2022-05-12 02:45:53,136 INFO: Train Loss:0.276 | Acc:0.9316 | F1:0.6643
    2022-05-12 02:46:03,772 INFO: val Loss:0.257 | Acc:0.9333 | F1:0.6248
    2022-05-12 02:46:05,667 INFO: -----------------SAVE:28epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.83it/s]
    2022-05-12 02:48:02,677 INFO: Epoch:[029/200]
    2022-05-12 02:48:02,678 INFO: Train Loss:0.301 | Acc:0.9240 | F1:0.6336
    2022-05-12 02:48:13,118 INFO: val Loss:0.231 | Acc:0.9368 | F1:0.6688
    2022-05-12 02:48:15,087 INFO: -----------------SAVE:29epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.83it/s]
    2022-05-12 02:50:12,140 INFO: Epoch:[030/200]
    2022-05-12 02:50:12,140 INFO: Train Loss:0.257 | Acc:0.9348 | F1:0.6902
    2022-05-12 02:50:22,854 INFO: val Loss:0.256 | Acc:0.9298 | F1:0.6502
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.83it/s]
    2022-05-12 02:52:20,020 INFO: Epoch:[031/200]
    2022-05-12 02:52:20,020 INFO: Train Loss:0.280 | Acc:0.9275 | F1:0.6676
    2022-05-12 02:52:32,650 INFO: val Loss:0.224 | Acc:0.9357 | F1:0.6491
    2022-05-12 02:52:34,529 INFO: -----------------SAVE:31epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.78it/s]
    2022-05-12 02:54:34,672 INFO: Epoch:[032/200]
    2022-05-12 02:54:34,672 INFO: Train Loss:0.258 | Acc:0.9313 | F1:0.6761
    2022-05-12 02:54:45,449 INFO: val Loss:0.233 | Acc:0.9404 | F1:0.6653
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:02<00:00,  1.74it/s]
    2022-05-12 02:56:48,418 INFO: Epoch:[033/200]
    2022-05-12 02:56:48,419 INFO: Train Loss:0.230 | Acc:0.9369 | F1:0.7012
    2022-05-12 02:56:59,769 INFO: val Loss:0.221 | Acc:0.9392 | F1:0.6830
    2022-05-12 02:57:01,715 INFO: -----------------SAVE:33epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.77it/s]
    2022-05-12 02:59:02,547 INFO: Epoch:[034/200]
    2022-05-12 02:59:02,548 INFO: Train Loss:0.238 | Acc:0.9363 | F1:0.7063
    2022-05-12 02:59:13,603 INFO: val Loss:0.244 | Acc:0.9462 | F1:0.6958
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.78it/s]
    2022-05-12 03:01:14,156 INFO: Epoch:[035/200]
    2022-05-12 03:01:14,157 INFO: Train Loss:0.246 | Acc:0.9375 | F1:0.7075
    2022-05-12 03:01:25,294 INFO: val Loss:0.248 | Acc:0.9392 | F1:0.6816
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.78it/s]
    2022-05-12 03:03:25,527 INFO: Epoch:[036/200]
    2022-05-12 03:03:25,528 INFO: Train Loss:0.224 | Acc:0.9427 | F1:0.7488
    2022-05-12 03:03:36,545 INFO: val Loss:0.225 | Acc:0.9439 | F1:0.7209
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.77it/s]
    2022-05-12 03:05:37,163 INFO: Epoch:[037/200]
    2022-05-12 03:05:37,163 INFO: Train Loss:0.213 | Acc:0.9462 | F1:0.7648
    2022-05-12 03:05:48,162 INFO: val Loss:0.268 | Acc:0.9357 | F1:0.6337
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.77it/s]
    2022-05-12 03:07:48,953 INFO: Epoch:[038/200]
    2022-05-12 03:07:48,954 INFO: Train Loss:0.214 | Acc:0.9401 | F1:0.7351
    2022-05-12 03:07:59,973 INFO: val Loss:0.223 | Acc:0.9392 | F1:0.6977
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.78it/s]
    2022-05-12 03:10:00,322 INFO: Epoch:[039/200]
    2022-05-12 03:10:00,322 INFO: Train Loss:0.213 | Acc:0.9421 | F1:0.7459
    2022-05-12 03:10:11,446 INFO: val Loss:0.179 | Acc:0.9520 | F1:0.7379
    2022-05-12 03:10:13,356 INFO: -----------------SAVE:39epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.78it/s]
    2022-05-12 03:12:13,660 INFO: Epoch:[040/200]
    2022-05-12 03:12:13,661 INFO: Train Loss:0.211 | Acc:0.9454 | F1:0.7598
    2022-05-12 03:12:24,828 INFO: val Loss:0.243 | Acc:0.9333 | F1:0.6770
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:01<00:00,  1.76it/s]
    2022-05-12 03:14:26,618 INFO: Epoch:[041/200]
    2022-05-12 03:14:26,618 INFO: Train Loss:0.213 | Acc:0.9439 | F1:0.7544
    2022-05-12 03:14:37,640 INFO: val Loss:0.217 | Acc:0.9404 | F1:0.7001
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.77it/s]
    2022-05-12 03:16:38,281 INFO: Epoch:[042/200]
    2022-05-12 03:16:38,281 INFO: Train Loss:0.185 | Acc:0.9474 | F1:0.7649
    2022-05-12 03:16:49,351 INFO: val Loss:0.293 | Acc:0.9275 | F1:0.7176
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:01<00:00,  1.77it/s]
    2022-05-12 03:18:50,512 INFO: Epoch:[043/200]
    2022-05-12 03:18:50,512 INFO: Train Loss:0.205 | Acc:0.9471 | F1:0.7788
    2022-05-12 03:19:01,619 INFO: val Loss:0.218 | Acc:0.9450 | F1:0.7089
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:01<00:00,  1.77it/s]
    2022-05-12 03:21:02,730 INFO: Epoch:[044/200]
    2022-05-12 03:21:02,731 INFO: Train Loss:0.200 | Acc:0.9503 | F1:0.7963
    2022-05-12 03:21:13,777 INFO: val Loss:0.241 | Acc:0.9427 | F1:0.6849
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.77it/s]
    2022-05-12 03:23:14,615 INFO: Epoch:[045/200]
    2022-05-12 03:23:14,616 INFO: Train Loss:0.206 | Acc:0.9439 | F1:0.7630
    2022-05-12 03:23:25,593 INFO: val Loss:0.215 | Acc:0.9497 | F1:0.7255
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:00<00:00,  1.78it/s]
    2022-05-12 03:25:25,830 INFO: Epoch:[046/200]
    2022-05-12 03:25:25,830 INFO: Train Loss:0.192 | Acc:0.9506 | F1:0.7924
    2022-05-12 03:25:36,350 INFO: val Loss:0.304 | Acc:0.9415 | F1:0.6865
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 03:27:34,020 INFO: Epoch:[047/200]
    2022-05-12 03:27:34,021 INFO: Train Loss:0.205 | Acc:0.9456 | F1:0.7732
    2022-05-12 03:27:44,488 INFO: val Loss:0.243 | Acc:0.9345 | F1:0.6825
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 03:29:42,032 INFO: Epoch:[048/200]
    2022-05-12 03:29:42,033 INFO: Train Loss:0.189 | Acc:0.9489 | F1:0.7791
    2022-05-12 03:29:52,487 INFO: val Loss:0.248 | Acc:0.9392 | F1:0.7005
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 03:31:49,840 INFO: Epoch:[049/200]
    2022-05-12 03:31:49,840 INFO: Train Loss:0.182 | Acc:0.9532 | F1:0.8132
    2022-05-12 03:32:00,388 INFO: val Loss:0.240 | Acc:0.9462 | F1:0.7128
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 03:33:57,795 INFO: Epoch:[050/200]
    2022-05-12 03:33:57,795 INFO: Train Loss:0.182 | Acc:0.9468 | F1:0.7743
    2022-05-12 03:34:08,363 INFO: val Loss:0.270 | Acc:0.9064 | F1:0.7416
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-12 03:36:07,963 INFO: Epoch:[051/200]
    2022-05-12 03:36:07,963 INFO: Train Loss:0.189 | Acc:0.9503 | F1:0.7811
    2022-05-12 03:36:18,447 INFO: val Loss:0.585 | Acc:0.8795 | F1:0.6918
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.83it/s]
    2022-05-12 03:38:15,680 INFO: Epoch:[052/200]
    2022-05-12 03:38:15,681 INFO: Train Loss:0.185 | Acc:0.9512 | F1:0.8051
    2022-05-12 03:38:26,392 INFO: val Loss:0.222 | Acc:0.9485 | F1:0.7017
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.83it/s]
    2022-05-12 03:40:23,467 INFO: Epoch:[053/200]
    2022-05-12 03:40:23,468 INFO: Train Loss:0.160 | Acc:0.9573 | F1:0.8349
    2022-05-12 03:40:34,065 INFO: val Loss:0.233 | Acc:0.9532 | F1:0.7353
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 03:42:31,345 INFO: Epoch:[054/200]
    2022-05-12 03:42:31,345 INFO: Train Loss:0.177 | Acc:0.9550 | F1:0.8194
    2022-05-12 03:42:41,879 INFO: val Loss:0.226 | Acc:0.9427 | F1:0.7080
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 03:44:39,316 INFO: Epoch:[055/200]
    2022-05-12 03:44:39,317 INFO: Train Loss:0.179 | Acc:0.9541 | F1:0.8264
    2022-05-12 03:44:49,779 INFO: val Loss:0.202 | Acc:0.9544 | F1:0.7468
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 03:46:47,307 INFO: Epoch:[056/200]
    2022-05-12 03:46:47,307 INFO: Train Loss:0.167 | Acc:0.9547 | F1:0.8249
    2022-05-12 03:46:57,850 INFO: val Loss:0.228 | Acc:0.9485 | F1:0.7131
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 03:48:55,567 INFO: Epoch:[057/200]
    2022-05-12 03:48:55,568 INFO: Train Loss:0.179 | Acc:0.9506 | F1:0.8015
    2022-05-12 03:49:06,109 INFO: val Loss:0.167 | Acc:0.9567 | F1:0.7839
    2022-05-12 03:49:08,412 INFO: -----------------SAVE:57epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-12 03:51:07,304 INFO: Epoch:[058/200]
    2022-05-12 03:51:07,304 INFO: Train Loss:0.167 | Acc:0.9538 | F1:0.8254
    2022-05-12 03:51:17,819 INFO: val Loss:0.230 | Acc:0.9427 | F1:0.6904
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-12 03:53:15,852 INFO: Epoch:[059/200]
    2022-05-12 03:53:15,852 INFO: Train Loss:0.176 | Acc:0.9524 | F1:0.8086
    2022-05-12 03:53:26,263 INFO: val Loss:0.262 | Acc:0.9298 | F1:0.7357
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 03:55:23,600 INFO: Epoch:[060/200]
    2022-05-12 03:55:23,600 INFO: Train Loss:0.164 | Acc:0.9532 | F1:0.8046
    2022-05-12 03:55:34,149 INFO: val Loss:0.251 | Acc:0.9474 | F1:0.7218
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 03:57:31,531 INFO: Epoch:[061/200]
    2022-05-12 03:57:31,531 INFO: Train Loss:0.163 | Acc:0.9559 | F1:0.8293
    2022-05-12 03:57:42,062 INFO: val Loss:0.223 | Acc:0.9427 | F1:0.6899
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.81it/s]
    2022-05-12 03:59:40,003 INFO: Epoch:[062/200]
    2022-05-12 03:59:40,004 INFO: Train Loss:0.150 | Acc:0.9559 | F1:0.8261
    2022-05-12 03:59:51,114 INFO: val Loss:0.214 | Acc:0.9450 | F1:0.6853
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 04:01:48,506 INFO: Epoch:[063/200]
    2022-05-12 04:01:48,506 INFO: Train Loss:0.176 | Acc:0.9509 | F1:0.7873
    2022-05-12 04:01:58,981 INFO: val Loss:0.216 | Acc:0.9427 | F1:0.6907
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 04:03:56,312 INFO: Epoch:[064/200]
    2022-05-12 04:03:56,312 INFO: Train Loss:0.149 | Acc:0.9603 | F1:0.8314
    2022-05-12 04:04:06,876 INFO: val Loss:0.281 | Acc:0.9333 | F1:0.6981
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.83it/s]
    2022-05-12 04:06:04,133 INFO: Epoch:[065/200]
    2022-05-12 04:06:04,134 INFO: Train Loss:0.146 | Acc:0.9585 | F1:0.8276
    2022-05-12 04:06:14,649 INFO: val Loss:0.239 | Acc:0.9450 | F1:0.7035
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:07<00:00,  1.68it/s]
    2022-05-12 04:08:21,848 INFO: Epoch:[066/200]
    2022-05-12 04:08:21,848 INFO: Train Loss:0.151 | Acc:0.9591 | F1:0.8407
    2022-05-12 04:08:35,494 INFO: val Loss:0.242 | Acc:0.9333 | F1:0.7182
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:01<00:00,  1.76it/s]
    2022-05-12 04:10:37,382 INFO: Epoch:[067/200]
    2022-05-12 04:10:37,383 INFO: Train Loss:0.142 | Acc:0.9597 | F1:0.8418
    2022-05-12 04:10:47,968 INFO: val Loss:0.206 | Acc:0.9474 | F1:0.7634
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 04:12:45,743 INFO: Epoch:[068/200]
    2022-05-12 04:12:45,743 INFO: Train Loss:0.149 | Acc:0.9565 | F1:0.8285
    2022-05-12 04:12:56,287 INFO: val Loss:0.195 | Acc:0.9520 | F1:0.7410
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-12 04:14:54,956 INFO: Epoch:[069/200]
    2022-05-12 04:14:54,956 INFO: Train Loss:0.136 | Acc:0.9614 | F1:0.8563
    2022-05-12 04:15:06,392 INFO: val Loss:0.211 | Acc:0.9439 | F1:0.7451
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.81it/s]
    2022-05-12 04:17:04,352 INFO: Epoch:[070/200]
    2022-05-12 04:17:04,352 INFO: Train Loss:0.134 | Acc:0.9641 | F1:0.8715
    2022-05-12 04:17:14,795 INFO: val Loss:0.201 | Acc:0.9520 | F1:0.7362
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 04:19:12,129 INFO: Epoch:[071/200]
    2022-05-12 04:19:12,129 INFO: Train Loss:0.126 | Acc:0.9655 | F1:0.8680
    2022-05-12 04:19:22,595 INFO: val Loss:0.241 | Acc:0.9509 | F1:0.7235
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.83it/s]
    2022-05-12 04:21:19,727 INFO: Epoch:[072/200]
    2022-05-12 04:21:19,728 INFO: Train Loss:0.138 | Acc:0.9641 | F1:0.8546
    2022-05-12 04:21:30,146 INFO: val Loss:0.247 | Acc:0.9392 | F1:0.7513
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.81it/s]
    2022-05-12 04:23:28,078 INFO: Epoch:[073/200]
    2022-05-12 04:23:28,079 INFO: Train Loss:0.139 | Acc:0.9635 | F1:0.8641
    2022-05-12 04:23:38,525 INFO: val Loss:0.222 | Acc:0.9520 | F1:0.7128
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 04:25:35,821 INFO: Epoch:[074/200]
    2022-05-12 04:25:35,821 INFO: Train Loss:0.141 | Acc:0.9629 | F1:0.8449
    2022-05-12 04:25:46,323 INFO: val Loss:0.199 | Acc:0.9439 | F1:0.7500
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-12 04:27:45,208 INFO: Epoch:[075/200]
    2022-05-12 04:27:45,208 INFO: Train Loss:0.132 | Acc:0.9632 | F1:0.8576
    2022-05-12 04:27:55,750 INFO: val Loss:0.201 | Acc:0.9520 | F1:0.7586
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.83it/s]
    2022-05-12 04:29:53,002 INFO: Epoch:[076/200]
    2022-05-12 04:29:53,003 INFO: Train Loss:0.123 | Acc:0.9693 | F1:0.8899
    2022-05-12 04:30:03,676 INFO: val Loss:0.195 | Acc:0.9509 | F1:0.7898
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-12 04:32:02,969 INFO: Epoch:[077/200]
    2022-05-12 04:32:02,969 INFO: Train Loss:0.138 | Acc:0.9623 | F1:0.8585
    2022-05-12 04:32:13,911 INFO: val Loss:0.177 | Acc:0.9637 | F1:0.8065
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.79it/s]
    2022-05-12 04:34:13,776 INFO: Epoch:[078/200]
    2022-05-12 04:34:13,777 INFO: Train Loss:0.119 | Acc:0.9702 | F1:0.8875
    2022-05-12 04:34:24,773 INFO: val Loss:0.174 | Acc:0.9591 | F1:0.7905
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [02:02<00:00,  1.75it/s]
    2022-05-12 04:36:27,041 INFO: Epoch:[079/200]
    2022-05-12 04:36:27,041 INFO: Train Loss:0.131 | Acc:0.9641 | F1:0.8578
    2022-05-12 04:36:37,896 INFO: val Loss:0.162 | Acc:0.9614 | F1:0.8143
    2022-05-12 04:36:40,131 INFO: -----------------SAVE:79epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 04:38:37,513 INFO: Epoch:[080/200]
    2022-05-12 04:38:37,513 INFO: Train Loss:0.132 | Acc:0.9629 | F1:0.8609
    2022-05-12 04:38:48,001 INFO: val Loss:0.182 | Acc:0.9579 | F1:0.7711
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 04:40:45,550 INFO: Epoch:[081/200]
    2022-05-12 04:40:45,550 INFO: Train Loss:0.119 | Acc:0.9696 | F1:0.8898
    2022-05-12 04:40:56,036 INFO: val Loss:0.191 | Acc:0.9579 | F1:0.7691
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.83it/s]
    2022-05-12 04:42:53,257 INFO: Epoch:[082/200]
    2022-05-12 04:42:53,257 INFO: Train Loss:0.122 | Acc:0.9667 | F1:0.8672
    2022-05-12 04:43:03,961 INFO: val Loss:0.188 | Acc:0.9567 | F1:0.7473
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.83it/s]
    2022-05-12 04:45:00,968 INFO: Epoch:[083/200]
    2022-05-12 04:45:00,968 INFO: Train Loss:0.114 | Acc:0.9664 | F1:0.8600
    2022-05-12 04:45:11,636 INFO: val Loss:0.173 | Acc:0.9591 | F1:0.7880
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 04:47:09,184 INFO: Epoch:[084/200]
    2022-05-12 04:47:09,185 INFO: Train Loss:0.119 | Acc:0.9673 | F1:0.8794
    2022-05-12 04:47:19,768 INFO: val Loss:0.279 | Acc:0.9287 | F1:0.7182
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 04:49:17,284 INFO: Epoch:[085/200]
    2022-05-12 04:49:17,285 INFO: Train Loss:0.115 | Acc:0.9705 | F1:0.8900
    2022-05-12 04:49:27,798 INFO: val Loss:0.196 | Acc:0.9591 | F1:0.7919
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 04:51:25,469 INFO: Epoch:[086/200]
    2022-05-12 04:51:25,469 INFO: Train Loss:0.114 | Acc:0.9699 | F1:0.8881
    2022-05-12 04:51:35,964 INFO: val Loss:0.223 | Acc:0.9462 | F1:0.7464
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.81it/s]
    2022-05-12 04:53:33,917 INFO: Epoch:[087/200]
    2022-05-12 04:53:33,917 INFO: Train Loss:0.107 | Acc:0.9714 | F1:0.8927
    2022-05-12 04:53:44,414 INFO: val Loss:0.236 | Acc:0.9544 | F1:0.7451
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 04:55:42,140 INFO: Epoch:[088/200]
    2022-05-12 04:55:42,141 INFO: Train Loss:0.111 | Acc:0.9693 | F1:0.8888
    2022-05-12 04:55:52,625 INFO: val Loss:0.188 | Acc:0.9649 | F1:0.7923
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 04:57:50,261 INFO: Epoch:[089/200]
    2022-05-12 04:57:50,262 INFO: Train Loss:0.104 | Acc:0.9717 | F1:0.8999
    2022-05-12 04:58:00,780 INFO: val Loss:0.199 | Acc:0.9579 | F1:0.7821
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-12 04:59:59,420 INFO: Epoch:[090/200]
    2022-05-12 04:59:59,421 INFO: Train Loss:0.117 | Acc:0.9708 | F1:0.8956
    2022-05-12 05:00:10,059 INFO: val Loss:0.160 | Acc:0.9661 | F1:0.8329
    2022-05-12 05:00:11,910 INFO: -----------------SAVE:90epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 05:02:09,608 INFO: Epoch:[091/200]
    2022-05-12 05:02:09,608 INFO: Train Loss:0.115 | Acc:0.9690 | F1:0.8826
    2022-05-12 05:02:20,112 INFO: val Loss:0.207 | Acc:0.9602 | F1:0.7957
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.80it/s]
    2022-05-12 05:04:18,795 INFO: Epoch:[092/200]
    2022-05-12 05:04:18,795 INFO: Train Loss:0.097 | Acc:0.9705 | F1:0.8934
    2022-05-12 05:04:29,165 INFO: val Loss:0.180 | Acc:0.9614 | F1:0.7952
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 05:06:27,033 INFO: Epoch:[093/200]
    2022-05-12 05:06:27,033 INFO: Train Loss:0.107 | Acc:0.9702 | F1:0.8988
    2022-05-12 05:06:37,567 INFO: val Loss:0.172 | Acc:0.9649 | F1:0.7973
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:58<00:00,  1.81it/s]
    2022-05-12 05:08:35,935 INFO: Epoch:[094/200]
    2022-05-12 05:08:35,936 INFO: Train Loss:0.099 | Acc:0.9719 | F1:0.9022
    2022-05-12 05:08:47,984 INFO: val Loss:0.185 | Acc:0.9579 | F1:0.7894
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:59<00:00,  1.80it/s]
    2022-05-12 05:10:47,130 INFO: Epoch:[095/200]
    2022-05-12 05:10:47,131 INFO: Train Loss:0.106 | Acc:0.9696 | F1:0.8754
    2022-05-12 05:10:57,497 INFO: val Loss:0.175 | Acc:0.9532 | F1:0.7878
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 05:12:53,948 INFO: Epoch:[096/200]
    2022-05-12 05:12:53,949 INFO: Train Loss:0.113 | Acc:0.9687 | F1:0.8867
    2022-05-12 05:13:04,259 INFO: val Loss:0.242 | Acc:0.9439 | F1:0.7562
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 05:15:00,755 INFO: Epoch:[097/200]
    2022-05-12 05:15:00,755 INFO: Train Loss:0.097 | Acc:0.9711 | F1:0.8894
    2022-05-12 05:15:11,110 INFO: val Loss:0.165 | Acc:0.9637 | F1:0.7994
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 05:17:07,287 INFO: Epoch:[098/200]
    2022-05-12 05:17:07,287 INFO: Train Loss:0.086 | Acc:0.9746 | F1:0.8990
    2022-05-12 05:17:17,643 INFO: val Loss:0.221 | Acc:0.9544 | F1:0.7789
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 05:19:13,708 INFO: Epoch:[099/200]
    2022-05-12 05:19:13,709 INFO: Train Loss:0.094 | Acc:0.9725 | F1:0.8937
    2022-05-12 05:19:24,079 INFO: val Loss:0.208 | Acc:0.9602 | F1:0.7956
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 05:21:20,349 INFO: Epoch:[100/200]
    2022-05-12 05:21:20,350 INFO: Train Loss:0.079 | Acc:0.9775 | F1:0.9179
    2022-05-12 05:21:30,748 INFO: val Loss:0.178 | Acc:0.9614 | F1:0.7887
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 05:23:27,501 INFO: Epoch:[101/200]
    2022-05-12 05:23:27,502 INFO: Train Loss:0.109 | Acc:0.9728 | F1:0.8853
    2022-05-12 05:23:37,919 INFO: val Loss:0.193 | Acc:0.9567 | F1:0.7863
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 05:25:34,697 INFO: Epoch:[102/200]
    2022-05-12 05:25:34,698 INFO: Train Loss:0.083 | Acc:0.9749 | F1:0.9035
    2022-05-12 05:25:45,119 INFO: val Loss:0.229 | Acc:0.9427 | F1:0.7955
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 05:27:41,871 INFO: Epoch:[103/200]
    2022-05-12 05:27:41,872 INFO: Train Loss:0.090 | Acc:0.9743 | F1:0.9066
    2022-05-12 05:27:52,311 INFO: val Loss:0.182 | Acc:0.9591 | F1:0.7741
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 05:29:49,965 INFO: Epoch:[104/200]
    2022-05-12 05:29:49,965 INFO: Train Loss:0.076 | Acc:0.9763 | F1:0.9077
    2022-05-12 05:30:00,578 INFO: val Loss:0.179 | Acc:0.9673 | F1:0.8121
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.83it/s]
    2022-05-12 05:31:57,621 INFO: Epoch:[105/200]
    2022-05-12 05:31:57,621 INFO: Train Loss:0.086 | Acc:0.9769 | F1:0.9095
    2022-05-12 05:32:07,990 INFO: val Loss:0.160 | Acc:0.9673 | F1:0.8194
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 05:34:04,848 INFO: Epoch:[106/200]
    2022-05-12 05:34:04,848 INFO: Train Loss:0.090 | Acc:0.9755 | F1:0.9119
    2022-05-12 05:34:15,184 INFO: val Loss:0.186 | Acc:0.9591 | F1:0.7916
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 05:36:11,775 INFO: Epoch:[107/200]
    2022-05-12 05:36:11,776 INFO: Train Loss:0.079 | Acc:0.9781 | F1:0.9262
    2022-05-12 05:36:22,099 INFO: val Loss:0.195 | Acc:0.9637 | F1:0.8185
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 05:38:18,867 INFO: Epoch:[108/200]
    2022-05-12 05:38:18,868 INFO: Train Loss:0.083 | Acc:0.9778 | F1:0.9279
    2022-05-12 05:38:29,215 INFO: val Loss:0.164 | Acc:0.9626 | F1:0.8338
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 05:40:26,156 INFO: Epoch:[109/200]
    2022-05-12 05:40:26,157 INFO: Train Loss:0.077 | Acc:0.9787 | F1:0.9260
    2022-05-12 05:40:36,555 INFO: val Loss:0.172 | Acc:0.9614 | F1:0.7909
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 05:42:32,978 INFO: Epoch:[110/200]
    2022-05-12 05:42:32,978 INFO: Train Loss:0.080 | Acc:0.9760 | F1:0.9168
    2022-05-12 05:42:43,424 INFO: val Loss:0.202 | Acc:0.9591 | F1:0.7613
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 05:44:40,071 INFO: Epoch:[111/200]
    2022-05-12 05:44:40,072 INFO: Train Loss:0.090 | Acc:0.9772 | F1:0.9217
    2022-05-12 05:44:50,479 INFO: val Loss:0.215 | Acc:0.9544 | F1:0.7882
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 05:46:47,427 INFO: Epoch:[112/200]
    2022-05-12 05:46:47,428 INFO: Train Loss:0.080 | Acc:0.9769 | F1:0.9093
    2022-05-12 05:46:57,912 INFO: val Loss:0.198 | Acc:0.9579 | F1:0.7568
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 05:48:54,641 INFO: Epoch:[113/200]
    2022-05-12 05:48:54,641 INFO: Train Loss:0.074 | Acc:0.9798 | F1:0.9267
    2022-05-12 05:49:04,985 INFO: val Loss:0.190 | Acc:0.9696 | F1:0.8246
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 05:51:01,737 INFO: Epoch:[114/200]
    2022-05-12 05:51:01,738 INFO: Train Loss:0.079 | Acc:0.9804 | F1:0.9278
    2022-05-12 05:51:12,151 INFO: val Loss:0.159 | Acc:0.9661 | F1:0.7878
    2022-05-12 05:51:14,015 INFO: -----------------SAVE:114epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 05:53:11,544 INFO: Epoch:[115/200]
    2022-05-12 05:53:11,544 INFO: Train Loss:0.069 | Acc:0.9819 | F1:0.9353
    2022-05-12 05:53:21,922 INFO: val Loss:0.203 | Acc:0.9556 | F1:0.7718
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 05:55:18,617 INFO: Epoch:[116/200]
    2022-05-12 05:55:18,617 INFO: Train Loss:0.082 | Acc:0.9775 | F1:0.9198
    2022-05-12 05:55:28,984 INFO: val Loss:0.341 | Acc:0.9439 | F1:0.8289
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 05:57:25,625 INFO: Epoch:[117/200]
    2022-05-12 05:57:25,625 INFO: Train Loss:0.069 | Acc:0.9836 | F1:0.9508
    2022-05-12 05:57:35,977 INFO: val Loss:0.142 | Acc:0.9684 | F1:0.8326
    2022-05-12 05:57:37,822 INFO: -----------------SAVE:117epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 05:59:34,592 INFO: Epoch:[118/200]
    2022-05-12 05:59:34,592 INFO: Train Loss:0.079 | Acc:0.9775 | F1:0.9179
    2022-05-12 05:59:44,942 INFO: val Loss:0.227 | Acc:0.9591 | F1:0.7859
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 06:01:41,907 INFO: Epoch:[119/200]
    2022-05-12 06:01:41,908 INFO: Train Loss:0.060 | Acc:0.9816 | F1:0.9318
    2022-05-12 06:01:52,286 INFO: val Loss:0.233 | Acc:0.9591 | F1:0.7759
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 06:03:49,026 INFO: Epoch:[120/200]
    2022-05-12 06:03:49,027 INFO: Train Loss:0.080 | Acc:0.9760 | F1:0.9125
    2022-05-12 06:03:59,496 INFO: val Loss:0.207 | Acc:0.9567 | F1:0.7575
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 06:05:56,148 INFO: Epoch:[121/200]
    2022-05-12 06:05:56,149 INFO: Train Loss:0.076 | Acc:0.9790 | F1:0.9287
    2022-05-12 06:06:06,743 INFO: val Loss:0.174 | Acc:0.9661 | F1:0.8058
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 06:08:03,421 INFO: Epoch:[122/200]
    2022-05-12 06:08:03,422 INFO: Train Loss:0.067 | Acc:0.9816 | F1:0.9337
    2022-05-12 06:08:13,834 INFO: val Loss:0.210 | Acc:0.9626 | F1:0.7845
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 06:10:10,263 INFO: Epoch:[123/200]
    2022-05-12 06:10:10,264 INFO: Train Loss:0.069 | Acc:0.9828 | F1:0.9433
    2022-05-12 06:10:20,701 INFO: val Loss:0.174 | Acc:0.9684 | F1:0.8265
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 06:12:17,390 INFO: Epoch:[124/200]
    2022-05-12 06:12:17,390 INFO: Train Loss:0.051 | Acc:0.9851 | F1:0.9534
    2022-05-12 06:12:27,776 INFO: val Loss:0.179 | Acc:0.9614 | F1:0.7911
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 06:14:24,706 INFO: Epoch:[125/200]
    2022-05-12 06:14:24,706 INFO: Train Loss:0.052 | Acc:0.9877 | F1:0.9629
    2022-05-12 06:14:35,059 INFO: val Loss:0.179 | Acc:0.9614 | F1:0.7822
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 06:16:31,779 INFO: Epoch:[126/200]
    2022-05-12 06:16:31,780 INFO: Train Loss:0.064 | Acc:0.9825 | F1:0.9429
    2022-05-12 06:16:42,169 INFO: val Loss:0.187 | Acc:0.9602 | F1:0.8199
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.83it/s]
    2022-05-12 06:18:39,312 INFO: Epoch:[127/200]
    2022-05-12 06:18:39,312 INFO: Train Loss:0.067 | Acc:0.9819 | F1:0.9408
    2022-05-12 06:18:49,681 INFO: val Loss:0.152 | Acc:0.9719 | F1:0.8504
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 06:20:46,556 INFO: Epoch:[128/200]
    2022-05-12 06:20:46,556 INFO: Train Loss:0.054 | Acc:0.9857 | F1:0.9540
    2022-05-12 06:20:56,860 INFO: val Loss:0.182 | Acc:0.9626 | F1:0.8221
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 06:22:53,262 INFO: Epoch:[129/200]
    2022-05-12 06:22:53,263 INFO: Train Loss:0.055 | Acc:0.9839 | F1:0.9414
    2022-05-12 06:23:03,662 INFO: val Loss:0.163 | Acc:0.9614 | F1:0.8169
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 06:25:00,374 INFO: Epoch:[130/200]
    2022-05-12 06:25:00,375 INFO: Train Loss:0.064 | Acc:0.9831 | F1:0.9414
    2022-05-12 06:25:10,780 INFO: val Loss:0.286 | Acc:0.9310 | F1:0.8173
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 06:27:07,656 INFO: Epoch:[131/200]
    2022-05-12 06:27:07,657 INFO: Train Loss:0.061 | Acc:0.9828 | F1:0.9347
    2022-05-12 06:27:18,231 INFO: val Loss:0.199 | Acc:0.9614 | F1:0.7778
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 06:29:15,502 INFO: Epoch:[132/200]
    2022-05-12 06:29:15,503 INFO: Train Loss:0.055 | Acc:0.9836 | F1:0.9394
    2022-05-12 06:29:25,880 INFO: val Loss:0.174 | Acc:0.9637 | F1:0.8015
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 06:31:22,380 INFO: Epoch:[133/200]
    2022-05-12 06:31:22,380 INFO: Train Loss:0.057 | Acc:0.9842 | F1:0.9411
    2022-05-12 06:31:32,786 INFO: val Loss:0.167 | Acc:0.9731 | F1:0.8650
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.83it/s]
    2022-05-12 06:33:30,011 INFO: Epoch:[134/200]
    2022-05-12 06:33:30,012 INFO: Train Loss:0.058 | Acc:0.9854 | F1:0.9484
    2022-05-12 06:33:40,406 INFO: val Loss:0.145 | Acc:0.9719 | F1:0.8406
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 06:35:36,807 INFO: Epoch:[135/200]
    2022-05-12 06:35:36,808 INFO: Train Loss:0.051 | Acc:0.9836 | F1:0.9458
    2022-05-12 06:35:47,184 INFO: val Loss:0.157 | Acc:0.9649 | F1:0.8168
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 06:37:43,795 INFO: Epoch:[136/200]
    2022-05-12 06:37:43,795 INFO: Train Loss:0.057 | Acc:0.9816 | F1:0.9380
    2022-05-12 06:37:54,210 INFO: val Loss:0.152 | Acc:0.9637 | F1:0.8118
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 06:39:50,785 INFO: Epoch:[137/200]
    2022-05-12 06:39:50,785 INFO: Train Loss:0.060 | Acc:0.9848 | F1:0.9324
    2022-05-12 06:40:01,240 INFO: val Loss:0.172 | Acc:0.9614 | F1:0.7848
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 06:41:57,883 INFO: Epoch:[138/200]
    2022-05-12 06:41:57,884 INFO: Train Loss:0.043 | Acc:0.9883 | F1:0.9606
    2022-05-12 06:42:08,283 INFO: val Loss:0.160 | Acc:0.9649 | F1:0.8423
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 06:44:04,804 INFO: Epoch:[139/200]
    2022-05-12 06:44:04,804 INFO: Train Loss:0.068 | Acc:0.9848 | F1:0.9546
    2022-05-12 06:44:15,165 INFO: val Loss:0.177 | Acc:0.9637 | F1:0.7822
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 06:46:11,704 INFO: Epoch:[140/200]
    2022-05-12 06:46:11,704 INFO: Train Loss:0.038 | Acc:0.9904 | F1:0.9683
    2022-05-12 06:46:22,035 INFO: val Loss:0.131 | Acc:0.9708 | F1:0.8448
    2022-05-12 06:46:23,941 INFO: -----------------SAVE:140epoch----------------
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 06:48:20,474 INFO: Epoch:[141/200]
    2022-05-12 06:48:20,475 INFO: Train Loss:0.043 | Acc:0.9851 | F1:0.9448
    2022-05-12 06:48:30,832 INFO: val Loss:0.165 | Acc:0.9626 | F1:0.8126
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 06:50:27,356 INFO: Epoch:[142/200]
    2022-05-12 06:50:27,357 INFO: Train Loss:0.040 | Acc:0.9883 | F1:0.9599
    2022-05-12 06:50:37,773 INFO: val Loss:0.156 | Acc:0.9684 | F1:0.8339
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 06:52:34,382 INFO: Epoch:[143/200]
    2022-05-12 06:52:34,382 INFO: Train Loss:0.035 | Acc:0.9895 | F1:0.9657
    2022-05-12 06:52:44,692 INFO: val Loss:0.187 | Acc:0.9649 | F1:0.8065
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 06:54:41,697 INFO: Epoch:[144/200]
    2022-05-12 06:54:41,697 INFO: Train Loss:0.035 | Acc:0.9895 | F1:0.9666
    2022-05-12 06:54:52,065 INFO: val Loss:0.175 | Acc:0.9696 | F1:0.8419
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 06:56:48,577 INFO: Epoch:[145/200]
    2022-05-12 06:56:48,578 INFO: Train Loss:0.046 | Acc:0.9889 | F1:0.9582
    2022-05-12 06:56:58,945 INFO: val Loss:0.151 | Acc:0.9696 | F1:0.8496
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 06:58:55,563 INFO: Epoch:[146/200]
    2022-05-12 06:58:55,564 INFO: Train Loss:0.037 | Acc:0.9915 | F1:0.9728
    2022-05-12 06:59:05,944 INFO: val Loss:0.158 | Acc:0.9684 | F1:0.8422
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 07:01:02,308 INFO: Epoch:[147/200]
    2022-05-12 07:01:02,308 INFO: Train Loss:0.044 | Acc:0.9866 | F1:0.9518
    2022-05-12 07:01:12,656 INFO: val Loss:0.181 | Acc:0.9696 | F1:0.8402
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 07:03:08,970 INFO: Epoch:[148/200]
    2022-05-12 07:03:08,970 INFO: Train Loss:0.048 | Acc:0.9866 | F1:0.9562
    2022-05-12 07:03:19,281 INFO: val Loss:0.158 | Acc:0.9719 | F1:0.8563
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 07:05:15,713 INFO: Epoch:[149/200]
    2022-05-12 07:05:15,713 INFO: Train Loss:0.037 | Acc:0.9880 | F1:0.9574
    2022-05-12 07:05:26,144 INFO: val Loss:0.163 | Acc:0.9673 | F1:0.8508
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 07:07:22,459 INFO: Epoch:[150/200]
    2022-05-12 07:07:22,459 INFO: Train Loss:0.043 | Acc:0.9880 | F1:0.9592
    2022-05-12 07:07:32,896 INFO: val Loss:0.177 | Acc:0.9696 | F1:0.8353
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 07:09:29,176 INFO: Epoch:[151/200]
    2022-05-12 07:09:29,176 INFO: Train Loss:0.035 | Acc:0.9898 | F1:0.9665
    2022-05-12 07:09:39,608 INFO: val Loss:0.184 | Acc:0.9708 | F1:0.8399
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 07:11:36,408 INFO: Epoch:[152/200]
    2022-05-12 07:11:36,409 INFO: Train Loss:0.027 | Acc:0.9906 | F1:0.9638
    2022-05-12 07:11:46,791 INFO: val Loss:0.173 | Acc:0.9719 | F1:0.8520
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 07:13:43,383 INFO: Epoch:[153/200]
    2022-05-12 07:13:43,383 INFO: Train Loss:0.030 | Acc:0.9904 | F1:0.9667
    2022-05-12 07:13:53,797 INFO: val Loss:0.202 | Acc:0.9673 | F1:0.8383
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 07:15:50,387 INFO: Epoch:[154/200]
    2022-05-12 07:15:50,388 INFO: Train Loss:0.039 | Acc:0.9901 | F1:0.9656
    2022-05-12 07:16:00,900 INFO: val Loss:0.192 | Acc:0.9649 | F1:0.8190
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 07:17:57,500 INFO: Epoch:[155/200]
    2022-05-12 07:17:57,500 INFO: Train Loss:0.037 | Acc:0.9889 | F1:0.9656
    2022-05-12 07:18:07,876 INFO: val Loss:0.179 | Acc:0.9649 | F1:0.8174
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 07:20:04,750 INFO: Epoch:[156/200]
    2022-05-12 07:20:04,751 INFO: Train Loss:0.034 | Acc:0.9912 | F1:0.9726
    2022-05-12 07:20:15,103 INFO: val Loss:0.193 | Acc:0.9626 | F1:0.8213
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 07:22:11,451 INFO: Epoch:[157/200]
    2022-05-12 07:22:11,452 INFO: Train Loss:0.038 | Acc:0.9895 | F1:0.9628
    2022-05-12 07:22:21,777 INFO: val Loss:0.193 | Acc:0.9673 | F1:0.8275
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 07:24:18,736 INFO: Epoch:[158/200]
    2022-05-12 07:24:18,736 INFO: Train Loss:0.030 | Acc:0.9915 | F1:0.9727
    2022-05-12 07:24:29,155 INFO: val Loss:0.171 | Acc:0.9684 | F1:0.8479
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 07:26:25,973 INFO: Epoch:[159/200]
    2022-05-12 07:26:25,973 INFO: Train Loss:0.032 | Acc:0.9909 | F1:0.9702
    2022-05-12 07:26:36,302 INFO: val Loss:0.190 | Acc:0.9673 | F1:0.8303
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 07:28:33,150 INFO: Epoch:[160/200]
    2022-05-12 07:28:33,150 INFO: Train Loss:0.025 | Acc:0.9921 | F1:0.9735
    2022-05-12 07:28:43,540 INFO: val Loss:0.175 | Acc:0.9637 | F1:0.8231
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 07:30:40,381 INFO: Epoch:[161/200]
    2022-05-12 07:30:40,382 INFO: Train Loss:0.023 | Acc:0.9930 | F1:0.9746
    2022-05-12 07:30:50,749 INFO: val Loss:0.189 | Acc:0.9708 | F1:0.8362
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 07:32:47,542 INFO: Epoch:[162/200]
    2022-05-12 07:32:47,542 INFO: Train Loss:0.022 | Acc:0.9950 | F1:0.9840
    2022-05-12 07:32:57,962 INFO: val Loss:0.189 | Acc:0.9684 | F1:0.8638
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 07:34:54,760 INFO: Epoch:[163/200]
    2022-05-12 07:34:54,760 INFO: Train Loss:0.043 | Acc:0.9877 | F1:0.9581
    2022-05-12 07:35:05,188 INFO: val Loss:0.183 | Acc:0.9661 | F1:0.8426
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 07:37:02,040 INFO: Epoch:[164/200]
    2022-05-12 07:37:02,040 INFO: Train Loss:0.029 | Acc:0.9924 | F1:0.9731
    2022-05-12 07:37:12,447 INFO: val Loss:0.166 | Acc:0.9708 | F1:0.8525
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 07:39:09,221 INFO: Epoch:[165/200]
    2022-05-12 07:39:09,222 INFO: Train Loss:0.033 | Acc:0.9909 | F1:0.9665
    2022-05-12 07:39:19,657 INFO: val Loss:0.170 | Acc:0.9708 | F1:0.8519
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 07:41:16,300 INFO: Epoch:[166/200]
    2022-05-12 07:41:16,300 INFO: Train Loss:0.029 | Acc:0.9944 | F1:0.9842
    2022-05-12 07:41:26,726 INFO: val Loss:0.158 | Acc:0.9731 | F1:0.8728
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 07:43:23,166 INFO: Epoch:[167/200]
    2022-05-12 07:43:23,166 INFO: Train Loss:0.021 | Acc:0.9939 | F1:0.9811
    2022-05-12 07:43:33,559 INFO: val Loss:0.166 | Acc:0.9743 | F1:0.8584
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 07:45:30,154 INFO: Epoch:[168/200]
    2022-05-12 07:45:30,154 INFO: Train Loss:0.018 | Acc:0.9947 | F1:0.9829
    2022-05-12 07:45:40,624 INFO: val Loss:0.162 | Acc:0.9743 | F1:0.8673
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 07:47:37,472 INFO: Epoch:[169/200]
    2022-05-12 07:47:37,472 INFO: Train Loss:0.017 | Acc:0.9947 | F1:0.9822
    2022-05-12 07:47:47,778 INFO: val Loss:0.167 | Acc:0.9719 | F1:0.8558
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 07:49:44,273 INFO: Epoch:[170/200]
    2022-05-12 07:49:44,273 INFO: Train Loss:0.022 | Acc:0.9921 | F1:0.9725
    2022-05-12 07:49:54,674 INFO: val Loss:0.174 | Acc:0.9708 | F1:0.8348
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 07:51:51,260 INFO: Epoch:[171/200]
    2022-05-12 07:51:51,260 INFO: Train Loss:0.024 | Acc:0.9939 | F1:0.9817
    2022-05-12 07:52:01,668 INFO: val Loss:0.162 | Acc:0.9731 | F1:0.8716
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 07:53:58,250 INFO: Epoch:[172/200]
    2022-05-12 07:53:58,250 INFO: Train Loss:0.023 | Acc:0.9936 | F1:0.9736
    2022-05-12 07:54:08,599 INFO: val Loss:0.175 | Acc:0.9766 | F1:0.8663
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 07:56:05,081 INFO: Epoch:[173/200]
    2022-05-12 07:56:05,082 INFO: Train Loss:0.029 | Acc:0.9927 | F1:0.9762
    2022-05-12 07:56:15,516 INFO: val Loss:0.167 | Acc:0.9766 | F1:0.8626
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 07:58:11,848 INFO: Epoch:[174/200]
    2022-05-12 07:58:11,848 INFO: Train Loss:0.020 | Acc:0.9939 | F1:0.9788
    2022-05-12 07:58:22,145 INFO: val Loss:0.172 | Acc:0.9684 | F1:0.8179
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 08:00:18,844 INFO: Epoch:[175/200]
    2022-05-12 08:00:18,845 INFO: Train Loss:0.017 | Acc:0.9953 | F1:0.9856
    2022-05-12 08:00:29,200 INFO: val Loss:0.160 | Acc:0.9754 | F1:0.8619
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 08:02:25,963 INFO: Epoch:[176/200]
    2022-05-12 08:02:25,963 INFO: Train Loss:0.012 | Acc:0.9965 | F1:0.9861
    2022-05-12 08:02:36,267 INFO: val Loss:0.176 | Acc:0.9731 | F1:0.8504
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 08:04:32,679 INFO: Epoch:[177/200]
    2022-05-12 08:04:32,679 INFO: Train Loss:0.017 | Acc:0.9942 | F1:0.9812
    2022-05-12 08:04:43,044 INFO: val Loss:0.173 | Acc:0.9766 | F1:0.8737
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 08:06:40,041 INFO: Epoch:[178/200]
    2022-05-12 08:06:40,041 INFO: Train Loss:0.015 | Acc:0.9950 | F1:0.9828
    2022-05-12 08:06:50,422 INFO: val Loss:0.178 | Acc:0.9789 | F1:0.8706
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 08:08:46,931 INFO: Epoch:[179/200]
    2022-05-12 08:08:46,932 INFO: Train Loss:0.014 | Acc:0.9959 | F1:0.9856
    2022-05-12 08:08:57,284 INFO: val Loss:0.175 | Acc:0.9731 | F1:0.8530
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 08:10:54,092 INFO: Epoch:[180/200]
    2022-05-12 08:10:54,092 INFO: Train Loss:0.018 | Acc:0.9942 | F1:0.9803
    2022-05-12 08:11:04,657 INFO: val Loss:0.163 | Acc:0.9766 | F1:0.8701
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:57<00:00,  1.82it/s]
    2022-05-12 08:13:02,078 INFO: Epoch:[181/200]
    2022-05-12 08:13:02,078 INFO: Train Loss:0.012 | Acc:0.9962 | F1:0.9884
    2022-05-12 08:13:12,445 INFO: val Loss:0.162 | Acc:0.9766 | F1:0.8607
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 08:15:09,143 INFO: Epoch:[182/200]
    2022-05-12 08:15:09,144 INFO: Train Loss:0.015 | Acc:0.9950 | F1:0.9842
    2022-05-12 08:15:19,560 INFO: val Loss:0.161 | Acc:0.9766 | F1:0.8702
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 08:17:16,200 INFO: Epoch:[183/200]
    2022-05-12 08:17:16,200 INFO: Train Loss:0.010 | Acc:0.9959 | F1:0.9849
    2022-05-12 08:17:26,602 INFO: val Loss:0.177 | Acc:0.9731 | F1:0.8502
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 08:19:22,957 INFO: Epoch:[184/200]
    2022-05-12 08:19:22,958 INFO: Train Loss:0.020 | Acc:0.9956 | F1:0.9866
    2022-05-12 08:19:33,405 INFO: val Loss:0.179 | Acc:0.9754 | F1:0.8524
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 08:21:30,369 INFO: Epoch:[185/200]
    2022-05-12 08:21:30,369 INFO: Train Loss:0.012 | Acc:0.9965 | F1:0.9877
    2022-05-12 08:21:40,766 INFO: val Loss:0.165 | Acc:0.9766 | F1:0.8644
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 08:23:37,126 INFO: Epoch:[186/200]
    2022-05-12 08:23:37,126 INFO: Train Loss:0.013 | Acc:0.9965 | F1:0.9888
    2022-05-12 08:23:47,535 INFO: val Loss:0.169 | Acc:0.9719 | F1:0.8425
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 08:25:44,168 INFO: Epoch:[187/200]
    2022-05-12 08:25:44,168 INFO: Train Loss:0.014 | Acc:0.9956 | F1:0.9847
    2022-05-12 08:25:54,507 INFO: val Loss:0.173 | Acc:0.9743 | F1:0.8543
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 08:27:51,108 INFO: Epoch:[188/200]
    2022-05-12 08:27:51,108 INFO: Train Loss:0.008 | Acc:0.9985 | F1:0.9950
    2022-05-12 08:28:01,563 INFO: val Loss:0.171 | Acc:0.9743 | F1:0.8553
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.83it/s]
    2022-05-12 08:29:58,418 INFO: Epoch:[189/200]
    2022-05-12 08:29:58,419 INFO: Train Loss:0.012 | Acc:0.9968 | F1:0.9899
    2022-05-12 08:30:08,830 INFO: val Loss:0.167 | Acc:0.9766 | F1:0.8660
    100%|████████████████████████████████████████████████████████████████████████████████| 214/214 [01:56<00:00,  1.84it/s]
    2022-05-12 08:32:05,279 INFO: Epoch:[190/200]
    2022-05-12 08:32:05,280 INFO: Train Loss:0.014 | Acc:0.9956 | F1:0.9849
    2022-05-12 08:32:15,678 INFO: val Loss:0.169 | Acc:0.9778 | F1:0.8776
    2022-05-12 08:32:15,679 INFO: 
    Best Val Epoch:140 | Val Loss:0.1306 | Val Acc:0.9708 | Val F1:0.8448
    2022-05-12 08:32:15,679 INFO: Total Process time:408.250Minute
    


```python
models_path
```




    ['results/000', 'results/001', 'results/002', 'results/003', 'results/004']




```python
ensemble = ensemble_5fold(models_path, test_loader, device)
```

    100%|██████████████████████████████████████████████████████████████████████████████████| 34/34 [00:29<00:00,  1.16it/s]
    100%|██████████████████████████████████████████████████████████████████████████████████| 34/34 [00:25<00:00,  1.36it/s]
    100%|██████████████████████████████████████████████████████████████████████████████████| 34/34 [00:25<00:00,  1.34it/s]
    100%|██████████████████████████████████████████████████████████████████████████████████| 34/34 [00:25<00:00,  1.32it/s]
    100%|██████████████████████████████████████████████████████████████████████████████████| 34/34 [00:25<00:00,  1.34it/s]
    


```python
# For submission
sub.iloc[:, 1] = ensemble.argmax(axis=1)
labels = ['bottle-broken_large', 'bottle-broken_small', 'bottle-contamination', 'bottle-good', 'cable-bent_wire', 'cable-cable_swap', 'cable-combined', 'cable-cut_inner_insulation', 'cable-cut_outer_insulation', 'cable-good', 'cable-missing_cable', 'cable-missing_wire', 'cable-poke_insulation', 'capsule-crack', 'capsule-faulty_imprint', 'capsule-good', 'capsule-poke', 'capsule-scratch', 'capsule-squeeze', 'carpet-color', 'carpet-cut', 'carpet-good', 'carpet-hole', 'carpet-metal_contamination', 'carpet-thread', 'grid-bent', 'grid-broken', 'grid-glue', 'grid-good', 'grid-metal_contamination', 'grid-thread', 'hazelnut-crack', 'hazelnut-cut', 'hazelnut-good', 'hazelnut-hole', 'hazelnut-print', 'leather-color', 'leather-cut', 'leather-fold', 'leather-glue', 'leather-good', 'leather-poke', 'metal_nut-bent', 'metal_nut-color', 'metal_nut-flip', 'metal_nut-good', 'metal_nut-scratch', 'pill-color', 'pill-combined', 'pill-contamination', 'pill-crack', 'pill-faulty_imprint', 'pill-good', 'pill-pill_type', 'pill-scratch', 'screw-good', 'screw-manipulated_front', 'screw-scratch_head', 'screw-scratch_neck', 'screw-thread_side', 'screw-thread_top', 'tile-crack', 'tile-glue_strip', 'tile-good', 'tile-gray_stroke', 'tile-oil', 'tile-rough', 'toothbrush-defective', 'toothbrush-good', 'transistor-bent_lead', 'transistor-cut_lead', 'transistor-damaged_case', 'transistor-good', 'transistor-misplaced', 'wood-color', 'wood-combined', 'wood-good', 'wood-hole', 'wood-liquid', 'wood-scratch', 'zipper-broken_teeth', 'zipper-combined', 'zipper-fabric_border', 'zipper-fabric_interior', 'zipper-good', 'zipper-rough', 'zipper-split_teeth', 'zipper-squeezed_teeth']
original_labels = dict(zip(range(len(labels)),labels))
sub['label'] = sub['label'].replace(original_labels)
sub
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>tile-glue_strip</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>grid-good</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>transistor-good</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>tile-gray_stroke</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>tile-good</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2149</th>
      <td>2149</td>
      <td>tile-gray_stroke</td>
    </tr>
    <tr>
      <th>2150</th>
      <td>2150</td>
      <td>screw-good</td>
    </tr>
    <tr>
      <th>2151</th>
      <td>2151</td>
      <td>grid-good</td>
    </tr>
    <tr>
      <th>2152</th>
      <td>2152</td>
      <td>cable-good</td>
    </tr>
    <tr>
      <th>2153</th>
      <td>2153</td>
      <td>zipper-good</td>
    </tr>
  </tbody>
</table>
<p>2154 rows × 2 columns</p>
</div>




```python
sub.to_csv('./data/submission.csv', index=False)
```


```python
# 정상 샘플 개수
good_cnt = 0
for i in range(len(sub)):
    if sub['label'][i][-4:] == 'good':
        good_cnt += 1
print(good_cnt)
```

    1110
    


```python
# 학습에 사용한 모델의 batch_size, epoch, img_size, patience
print('batch_size =', args.batch_size)
print('epochs =', args.epochs)
print('img_size =', args.img_size)
print('patience =', args.patience)
```

    batch_size = 16
    epochs = 200
    img_size = 224
    patience = 50
    


```python
print('model =', args.encoder_name)
```

    model = regnety_160
    
