import os

import torch
import torch.utils.data as data
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader,Dataset
from torchvision import models as tmodels
import torchvision

from data_loader import FADADataset
import models
import data_loader
from losses import ContrastiveLoss, SpecLoss

src = 'herbarium'  #mnist
dst = 'photo'       #svhn
img_size = 224  #32
#datadir = '/home/villacis/Desktop/a4-code-v2-updated/minidigits'
datadir = '/home/villacis/Desktop/villacis/datasets/plantclef_superminida_17_20_split'
#datadir = '/home/villacis/Desktop/villacis/datasets/plantclef_minida_cropped'

num_epochs1 = 110
num_epochs2 = 300
num_epochs3 = 150


use_cuda=True if torch.cuda.is_available() else False
device=torch.device('cuda:0') if use_cuda else torch.device('cpu')


#encoder = models.Encoder()
encoder = tmodels.resnet18(pretrained=True)
#encoder = tmodels.inception_v3(pretrained=True)
#encoder.aux_logits=False
encoder.fc = nn.Sequential()
#discriminator = models.DCDPro(input_features=128)
decoder = models.Decoder()
encoder.to(device)
decoder = decoder.to(device)

loss_fn=torch.nn.MSELoss()
# -----------------------------------------------------------------------------
## etapa 1: entrenar g y h
print("||||| Stage 1 |||||")
optimizer=torch.optim.Adam(list(decoder.parameters()), lr = 0.0001)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)

data_transforms = {
    'train': transforms.Compose([
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        #transforms.Resize((img_size, img_size)),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

datasets_1 = {x: datasets.ImageFolder(os.path.join(datadir, dst, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders_1 = {x: torch.utils.data.DataLoader(datasets_1[x], batch_size=60,
                                             shuffle=True, num_workers=6)
              for x in ['train', 'val']}

print("Finished loading data")

encoder.load_state_dict(torch.load('result/encoder_base.pth'))
best_acc = 0.0
for i in range(num_epochs1):
    sum_loss = 0.0
    print("epoch ini {}".format(i+1))
    for data,labels in dataloaders_1['train']:
        if(data.size(0)<60):
            break
        data=data.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()
        a = encoder(data)
        pred = decoder(a)

        loss=loss_fn(pred, data)
        sum_loss+=loss.item()
        loss.backward()
        optimizer.step()
        torchvision.utils.save_image(data, "data_photo.png")
        torchvision.utils.save_image(pred, "pred_photo.png")
    scheduler.step()
    acc = 0
    print("Phase 1 ---- Epoch {} accuracy: {}".format((i+1), sum_loss))
    

