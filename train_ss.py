import os

import torch
import torch.utils.data as data
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader,Dataset
from torchvision import models as tmodels

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

num_epochs1 = 40
num_epochs2 = 300
num_epochs3 = 150


use_cuda=True if torch.cuda.is_available() else False
device=torch.device('cuda:0') if use_cuda else torch.device('cpu')


classifier = models.ClassifierPro()
#encoder = models.Encoder()
encoder = tmodels.resnet18(pretrained=True)
#encoder = tmodels.inception_v3(pretrained=True)
#encoder.aux_logits=False
encoder.fc = nn.Sequential()
discriminator = models.DCDPro()
#discriminator = models.DCDPro(input_features=128)
decoder = models.Decoder()

decoder.to(device)
classifier.to(device)
encoder.to(device)
discriminator.to(device)

loss_fn=torch.nn.CrossEntropyLoss()
loss_fn2 = ContrastiveLoss()  ##quitar
loss_fn3 = SpecLoss()  ##quitar
loss_fn4 = torch.nn.L1Loss()
# -----------------------------------------------------------------------------
## etapa 1: entrenar g y h
print("||||| Stage 1 |||||")
optimizer=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters())+list(decoder.parameters()), lr = 0.0001)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)

data_transforms = {
    'train': transforms.Compose([
        #transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(15),
        #transforms.RandomCrop((img_size, img_size)),
        #transforms.RandomResizedCrop((img_size, img_size)),
        transforms.CenterCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
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

datasets_1 = {x: datasets.ImageFolder(os.path.join(datadir, src, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders_1 = {x: torch.utils.data.DataLoader(datasets_1[x], batch_size=60,
                                             shuffle=True, num_workers=6)
              for x in ['train', 'val']}

print("Finished loading data")

best_acc = 0.0
for i in range(num_epochs1):
    for data,labels in dataloaders_1['train']:
        data=data.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()
        pred = classifier(encoder(data))
        rec = decoder(encoder(data))

        loss=loss_fn(pred,labels)
        loss_ss = loss_fn4(data, rec)
        loss = loss+loss_ss
        loss.backward()
        optimizer.step()
    scheduler.step()
    acc = 0
    for data, labels in dataloaders_1['val']:
        with torch.no_grad():
            data=data.to(device)
            labels=labels.to(device)
            y_test_pred=classifier(encoder(data))
            acc += (torch.max(y_test_pred,1)[1]==labels).float().mean().item()
    
    accuracy=round(acc / float(len(dataloaders_1['val'])), 3)
    
    if(accuracy>best_acc):
        best_acc = accuracy
        torch.save(encoder.state_dict(),'result/encoder_base.pth')
        classifier.save()
    print("Phase 1 ---- Epoch {} accuracy: {}".format((i+1), accuracy))
    
#encoder.save()
#torch.save(encoder.state_dict(),'result/encoder_base.pth')
#classifier.save()

#encoder.load()

encoder.load_state_dict(torch.load('result/encoder_base.pth'))
classifier.load()

# -----------------------------------------------------------------------------
## fase 2: entrenar dcd, congelar g y h
print("||||| Stage 2 |||||")

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001) #0.001

print("Carga optimizer")
#X_s,Y_s=data_loader.sample_src(os.path.join(datadir, src, 'train'), data_transforms['train'])
#X_t,Y_t=data_loader.sample_tgt(os.path.join(datadir, dst, 'train'), data_transforms['train'], n = 7)

siamese_dataset = FADADataset(os.path.join(datadir, src),
                                       os.path.join(datadir, dst),
                                        'train',
                                        img_size
                                        )
dataloader_2 = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=6,
                        batch_size=20)
print("Carga datos")


for epoch in range(num_epochs2):
    loss_mean = []
    cont = 0.0
    acc = 0.0
    buenos = 0.0
    confusion_matrix = torch.zeros(4, 4)
    for X1, X2, idx1, idx2, ground_truths, op1, op2, same in dataloader_2:
        cont += 1.0
        X1=X1.to(device)
        X2=X2.to(device)
        same = same.to(device)
        ground_truths=ground_truths.to(device)
        optimizer_D.zero_grad()
        X_cat=torch.cat([encoder(X1),encoder(X2)],1)
        y_pred=discriminator(X_cat.detach())
        #print(y_pred)
        #print(ground_truths)
        acc += (torch.max(y_pred,1)[1]==ground_truths).float().mean().item()
        pred = torch.max(y_pred,1)[1]
        for t, p in zip(ground_truths.view(-1), pred.view(-1)):
            cont += 1.0
            if(t == p):
                buenos += 1
            confusion_matrix[t.long(), p.long()] += 1
        ground_truths = ground_truths.view(-1)
        loss=loss_fn(y_pred,ground_truths)
        loss_contrastive = loss_fn2(encoder(X1),encoder(X2),same) ##quitar
        ###loss = loss + 0.004*loss_contrastive ##quitar
        loss.backward()
        optimizer_D.step()
        loss_mean.append(loss.item())
    #print(buenos)
    #print(cont)
    accuracy=round(buenos / cont, 3)
    print("Phase 2 ---- Epoch %d/%d loss:%.3f acc:%.3f"%(epoch+1,num_epochs2,np.mean(loss_mean), accuracy))
    print(confusion_matrix.diag()/confusion_matrix.sum(1))
    accuracy=round(buenos / cont, 3)

discriminator.save()

discriminator.load()

# -----------------------------------------------------------------------------
## fase 3: entrenar g y h, congelar dcd
print("||||| Stage 3 |||||")
optimizer_g_h=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters())+list(decoder.parameters()),lr=0.0001)  #0.0001
optimizer_d=torch.optim.Adam(discriminator.parameters(),lr=0.0001) #0.001
scheduler_g_h = lr_scheduler.MultiStepLR(optimizer_g_h, milestones=[135], gamma=0.1)
scheduler_d = lr_scheduler.MultiStepLR(optimizer_d, milestones=[135], gamma=0.1)

datasets_3 = {x: datasets.ImageFolder(os.path.join(datadir, dst, x),
                                          data_transforms[x])
                  for x in ['val']}
test_dataloader = torch.utils.data.DataLoader(datasets_3['val'], batch_size=10,
                                             shuffle=True, num_workers=6)            




acc = 0.0
for data, labels in dataloaders_1['val']:
    with torch.no_grad():
        data=data.to(device)
        labels=labels.to(device)
        y_test_pred=classifier(encoder(data))
        acc += (torch.max(y_test_pred,1)[1]==labels).float().mean().item()

accuracy=round(acc / float(len(dataloaders_1['val'])), 3)
print("Phase 1 ---- Epoch {} accuracy: {}".format((0), accuracy))

best_acc = 0.0
acc = 0.0
for data, labels in test_dataloader:
    data = data.to(device)
    labels = labels.to(device)
    y_test_pred = classifier(encoder(data))
    acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()

accuracy = round(acc / float(len(test_dataloader)), 3)
if(accuracy>best_acc):
    best_acc = accuracy
print("step3----Epoch %d/%d  accuracy: %.3f best_acc: %.3f " % (0, num_epochs3, accuracy, best_acc))



for epoch in range(num_epochs3):
    #---training g and h , DCD is frozen
    
    #----training g and h, dcd frozen
    cont = 0.0
    acc = 0.0
    buenos = 0.0
    confusion_matrix = torch.zeros(4, 4)
    for X1, X2, ground_truths_y1, ground_truths_y2, ground_truths, domains1, domains2, same in dataloader_2:
        X1=X1.to(device)
        X2=X2.to(device)
        ground_truths_y1=ground_truths_y1.to(device)
        ground_truths_y2 = ground_truths_y2.to(device)
        ground_truths=ground_truths.to(device)
        
        dcd_labels = ground_truths.view(-1)
        ground_truths_y1 = ground_truths_y1.view(-1)
        ground_truths_y1 = ground_truths_y1.view(-1)
        optimizer_g_h.zero_grad()

        encoder_X1=encoder(X1)
        encoder_X2=encoder(X2)
        
        rec_X1 = decoder(encoder_X1)
        rec_X2 = decoder(encoder_X2)

        X_cat=torch.cat([encoder_X1,encoder_X2],1)
        y_pred_X1=classifier(encoder_X1)
        y_pred_X2=classifier(encoder_X2)
        y_pred_dcd=discriminator(X_cat)
        
        pred = torch.max(y_pred_dcd,1)[1]
        for t, p in zip(ground_truths.view(-1), pred.view(-1)):
            cont += 1.0
            if(t == p):
                buenos += 1
            confusion_matrix[t.long(), p.long()] += 1

        loss_X1=loss_fn(y_pred_X1,ground_truths_y1)
        loss_X2=loss_fn(y_pred_X2,ground_truths_y2)
        loss_dcd=loss_fn(y_pred_dcd,dcd_labels)
        loss_ss = loss_fn4(rec_X1, X1) + loss_fn4(rec_X2, X2)
        #loss_contrastive = loss_fn2(encoder(X1),encoder(X2),same)
        
        nl11 = loss_fn3(y_pred_X1,ground_truths_y1, domains1, 1)
        nl12 = loss_fn3(y_pred_X1,ground_truths_y1, domains1, 2)
        
        nl21 = loss_fn3(y_pred_X2,ground_truths_y2, domains2, 1)
        nl22 = loss_fn3(y_pred_X2,ground_truths_y2, domains2, 2)
        #loss_X1 = nl11+nl21
        #loss_X2 = nl12+nl22
        
        #print(loss_X1+loss_X2)
        loss_sum = loss_X1 + loss_X2 - 0.9*loss_dcd + loss_ss

        loss_sum.backward()
        optimizer_g_h.step()
        
    #----training dcd ,g and h frozen
    for X1, X2, ground_truths_y1, ground_truths_y2, ground_truths, domains1, domains2, same in dataloader_2:
        X1 = X1.to(device)
        X2 = X2.to(device)
        same = same.to(device)
        ground_truths = ground_truths.to(device)
        ground_truths = ground_truths.view(-1)
        
        optimizer_d.zero_grad()
        X_cat = torch.cat([encoder(X1), encoder(X2)], 1)
        y_pred = discriminator(X_cat.detach())
        loss = loss_fn(y_pred, ground_truths)
        loss_contrastive = loss_fn2(encoder(X1),encoder(X2), same) ##quitar
        #loss = loss + loss_contrastive ##quitar
        loss.backward()
        optimizer_d.step()

    
    #testing
    acc = 0
    for data, labels in test_dataloader:
        data = data.to(device)
        labels = labels.to(device)
        y_test_pred = classifier(encoder(data))
        acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()

    accuracy = round(acc / float(len(test_dataloader)), 3)
    if(accuracy>best_acc):
        best_acc = accuracy
        #torch.save(encoder.state_dict(),'result/encoder.pth')
        #classifier.save()
        #discriminator.save()
    accuracy_dcd=round(buenos / cont, 3)
    print("step3----Epoch %d/%d  accuracy: %.3f best_acc: %.3f dcd_acc: %.3f" % (epoch + 1, num_epochs3, accuracy, best_acc, accuracy_dcd))
    print(confusion_matrix.diag()/confusion_matrix.sum(1))

acc = 0.0
for data, labels in test_dataloader:
    data = data.to(device)
    labels = labels.to(device)
    y_test_pred = classifier(encoder(data))
    acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()
accuracy = round(acc / float(len(test_dataloader)), 3)
print("step3----Epoch %d/%d  accuracy: %.3f best_acc: %.3f " % (0, num_epochs3, accuracy, best_acc))  


acc = 0.0
for data, labels in dataloaders_1['val']:
    with torch.no_grad():
        data=data.to(device)
        labels=labels.to(device)
        y_test_pred=classifier(encoder(data))
        acc += (torch.max(y_test_pred,1)[1]==labels).float().mean().item()

accuracy=round(acc / float(len(dataloaders_1['val'])), 3)
print("Phase 1 ---- Epoch {} accuracy: {}".format((0), accuracy))

"""


dataset_train = FADADataset(src_path = os.path.join(datadir, src),
                                dst_path = os.path.join(datadir, dst),
                                stage = 'train',
                                image_size = 32)
dataloader_train = DataLoader(dataset_train,
                        shuffle=True,
                        num_workers=4,
                        batch_size=64)

dataset_test = FADADataset(src_path = os.path.join(datadir, src),
                                dst_path = os.path.join(datadir, dst),
                                stage = 'test',
                                image_size = 32)
dataloader_test = DataLoader(dataset_test,
                        shuffle=True,
                        num_workers=4,
                        batch_size=64)

"""
