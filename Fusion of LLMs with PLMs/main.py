from DownstreamModel import DownstreamModel
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from model_op_new import Train, Test
import argparse
import os
import torch
from MyDataset import MyDataset
import json
from plot_loss import plot_loss
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--logs_dir', type=str, default='ASR_logs_20seeds')
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--SIGMA', type=float, default=0.3)  #[0.1,0.5]
parser.add_argument('--batch_size', type=int, default=4) 
parser.add_argument('--lr', type=float, default=1e-5) 
parser.add_argument('--model_name', type=str)
parser.add_argument('--version', type=str)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logs_dir=args.logs_dir
model_name=args.model_name
version=args.version
epoches = args.epoch
SIGMA = args.SIGMA
batch_size = args.batch_size
seed=args.seed
lr = args.lr
class_num = 2

seed_everything(seed)

current_log_path=f'{logs_dir}/{model_name}/{version}/{seed}'
Path(current_log_path).mkdir(parents=True, exist_ok=True)
f=open(f'{current_log_path}/train_log.txt','w')

l_dataset_path = f'llama3.1_embedding/ASR_trans/{model_name}/{version}/'
b_dataset_path = f'bert_embedding/fine_tuned_cls/ASR_trans/{model_name}/{version}/'
r_dataset_path = f'roberta_embedding/fine_tuned_cls/ASR_trans/{model_name}/{version}/'

mode = 'train'
train_data = MyDataset(mode, l_dataset_path, b_dataset_path, r_dataset_path)
l_sents_reps, b_sents_reps,r_sents_reps,labels = train_data[:10]
# print(f'l_sents_reps.shape: {l_sents_reps.shape}')
# print(f'b_sents_reps.shape: {b_sents_reps.shape}')
# print(f'r_sents_reps.shape: {r_sents_reps.shape}')
# print(f'labels.shape: {labels.shape}')
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
mode = 'test'
test_data = MyDataset(mode, l_dataset_path, b_dataset_path, r_dataset_path)   
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

model = DownstreamModel(class_num, SIGMA).to(device)

loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr)

print('training ...')
f.write('training ...\n')
for epoch in tqdm(range(epoches)):
    model = model.to(device)
    print(f'--------------------------- epoch {epoch+1} ---------------------------')
    print('training ...\n')
    f.write('training ...\n')
    Train(train_loader, device, model, loss_fn, optimizer,f)
    flag='train'
    Test(train_loader, device, model, loss_fn, f, flag)
    print('test ...\n')
    f.write('test ...\n')
    Test(test_loader, device, model, loss_fn,f)
f.close()
plot_loss(f'{current_log_path}/train_log.txt')