import os

import torch
import torch.utils.data as data
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import numpy as np
from torchvision import models as tmodels

#from data_loader import FADADataset
import models
import data_loader

src = 'mnist'  #mnist
dst = 'svhn'       #svhn
img_size = 32  #32
datadir = '/home/villacis/Desktop/a4-code-v2-updated/minidigits'
#datadir = '/home/villacis/Desktop/villacis/datasets/plantclef_minida_17_20_split'

num_epochs1 = 30
num_epochs2 = 800
num_epochs3 = 100


use_cuda=True if torch.cuda.is_available() else False
device=torch.device('cuda:0') if use_cuda else torch.device('cpu')


classifier = models.Classifier()
encoder = models.Encoder()
#encoder = tmodels.resnet18(pretrained=True)
#encoder = tmodels.inception_v3(pretrained=True)
#encoder.aux_logits=False
#encoder.fc = nn.Sequential()
#discriminator = models.DCDPro()
discriminator = models.DCD()
#discriminator = models.DCDPro(input_features=128)

classifier.to(device)
encoder.to(device)
discriminator.to(device)

loss_fn=torch.nn.CrossEntropyLoss()
# -----------------------------------------------------------------------------
## etapa 1: entrenar g y h
print("||||| Stage 1 |||||")
optimizer=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters()), lr = 0.0001)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[25], gamma=0.1)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((img_size, img_size)),
        #transforms.RandomCenterCrop((img_size, img_size)),
        #transforms.RandomRotation(15),
        #transforms.CenterCrop((img_size, img_size)),
        #transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize((img_size, img_size)),
        #transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

datasets_1 = {x: datasets.ImageFolder(os.path.join(datadir, src, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders_1 = {x: torch.utils.data.DataLoader(datasets_1[x], batch_size=60,
                                             shuffle=True, num_workers=6)
              for x in ['train', 'val']}

print("Finished loading data")
"""
for i in range(num_epochs1):
    for data,labels in dataloaders_1['train']:
        data=data.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()
        pred = classifier(encoder(data))

        loss=loss_fn(pred,labels)
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
    print("Phase 1 ---- Epoch {} accuracy: {}".format((i+1), accuracy))
    
#encoder.save()
torch.save(encoder.state_dict(),'result/encoder.pth')
classifier.save()
"""
#encoder.load()
encoder.load_state_dict(torch.load('result/encoder.pth'))
classifier.load()

# -----------------------------------------------------------------------------
## fase 2: entrenar dcd, congelar g y h
print("||||| Stage 2 |||||")

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001) #0.001

print("Carga optimizer")
X_s,Y_s=data_loader.sample_src(os.path.join(datadir, src, 'train'), data_transforms['train'])
X_t,Y_t=data_loader.sample_tgt(os.path.join(datadir, dst, 'train'), data_transforms['train'], n = 4)
print("Carga datos")


for epoch in range(num_epochs2):
    groups,aa = data_loader.sample_groups(X_s,Y_s,X_t,Y_t,seed=epoch)
    n_iters = 4 * len(groups[1])
    index_list = torch.randperm(n_iters)
    mini_batch_size=20 #use mini_batch train can be more stable
    loss_mean = []
    
    X1=[];X2=[];ground_truths=[]
    for index in range(n_iters):
        ground_truth=index_list[index]//len(groups[1])

        x1,x2=groups[ground_truth][index_list[index]-len(groups[1])*ground_truth]
        X1.append(x1)
        X2.append(x2)
        ground_truths.append(ground_truth)
        if (index+1)%mini_batch_size==0:
            X1=torch.stack(X1)
            X2=torch.stack(X2)
            ground_truths=torch.LongTensor(ground_truths)
            X1=X1.to(device)
            X2=X2.to(device)
            ground_truths=ground_truths.to(device)
            optimizer_D.zero_grad()
            X_cat=torch.cat([encoder(X1),encoder(X2)],1)
            y_pred=discriminator(X_cat.detach())
            loss=loss_fn(y_pred,ground_truths)
            loss.backward()
            optimizer_D.step()
            loss_mean.append(loss.item())
            X1 = []
            X2 = []
            ground_truths = []

    print("Phase 2 ---- Epoch %d/%d loss:%.3f"%(epoch+1,num_epochs2,np.mean(loss_mean)))


discriminator.save()

discriminator.load()

# -----------------------------------------------------------------------------
## fase 3: entrenar g y h, congelar dcd
print("||||| Stage 3 |||||")
optimizer_g_h=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters()),lr=0.001)  #0.0001
optimizer_d=torch.optim.Adam(discriminator.parameters(),lr=0.001) #0.001
scheduler_g_h = lr_scheduler.MultiStepLR(optimizer_g_h, milestones=[90], gamma=0.1)
scheduler_d = lr_scheduler.MultiStepLR(optimizer_d, milestones=[90], gamma=0.1)

datasets_3 = {x: datasets.ImageFolder(os.path.join(datadir, dst, x),
                                          data_transforms[x])
                  for x in ['val']}
test_dataloader = torch.utils.data.DataLoader(datasets_3['val'], batch_size=10,
                                             shuffle=True, num_workers=6)            




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





groups, groups_y = data_loader.sample_groups(X_s,Y_s,X_t,Y_t,seed=num_epochs2+epoch)

for epoch in range(num_epochs3):
    #---training g and h , DCD is frozen
    
    G1, G2, G3, G4 = groups
    Y1, Y2, Y3, Y4 = groups_y
    groups_2 = [G2, G4]
    groups_y_2 = [Y2, Y4]

    n_iters = 2 * len(G2)
    index_list = torch.randperm(n_iters)

    n_iters_dcd = 4 * len(G2)
    index_list_dcd = torch.randperm(n_iters_dcd)

    mini_batch_size_g_h = 15 #data only contains G2 and G4 ,so decrease mini_batch
    mini_batch_size_dcd= 15 #data contains G1,G2,G3,G4 so use 40 as mini_batch
    X1 = []
    X2 = []
    ground_truths_y1 = []
    ground_truths_y2 = []
    dcd_labels=[]
    
    
    for index in range(n_iters):


        ground_truth=index_list[index]//len(G2)
        x1, x2 = groups_2[ground_truth][index_list[index] - len(G2) * ground_truth]
        y1, y2 = groups_y_2[ground_truth][index_list[index] - len(G2) * ground_truth]
        #### y1=torch.LongTensor([y1.item()])
        ##### y2=torch.LongTensor([y2.item()])
        dcd_label=0 if ground_truth==0 else 2
        X1.append(x1)
        X2.append(x2)
        ground_truths_y1.append(y1)
        ground_truths_y2.append(y2)
        dcd_labels.append(dcd_label)
        if (index+1)%mini_batch_size_g_h==0:
            X1=torch.stack(X1)
            X2=torch.stack(X2)
            ground_truths_y1=torch.LongTensor(ground_truths_y1)
            ground_truths_y2 = torch.LongTensor(ground_truths_y2)
            dcd_labels=torch.LongTensor(dcd_labels)
            X1=X1.to(device)
            X2=X2.to(device)
            ground_truths_y1=ground_truths_y1.to(device)
            ground_truths_y2 = ground_truths_y2.to(device)
            dcd_labels=dcd_labels.to(device)

            optimizer_g_h.zero_grad()

            encoder_X1=encoder(X1)
            encoder_X2=encoder(X2)

            X_cat=torch.cat([encoder_X1,encoder_X2],1)
            y_pred_X1=classifier(encoder_X1)
            y_pred_X2=classifier(encoder_X2)
            y_pred_dcd=discriminator(X_cat)

            loss_X1=loss_fn(y_pred_X1,ground_truths_y1)
            loss_X2=loss_fn(y_pred_X2,ground_truths_y2)
            loss_dcd=loss_fn(y_pred_dcd,dcd_labels)

            loss_sum = loss_X1 + loss_X2 + 0.2 * loss_dcd  #0.5

            loss_sum.backward()
            optimizer_g_h.step()

            X1 = []
            X2 = []
            ground_truths_y1 = []
            ground_truths_y2 = []
            dcd_labels = []

    
    #----training dcd ,g and h frozen
    X1 = []
    X2 = []
    ground_truths = []
    for index in range(n_iters_dcd):

        ground_truth=index_list_dcd[index]//len(groups[1])

        x1, x2 = groups[ground_truth][index_list_dcd[index] - len(groups[1]) * ground_truth]
        X1.append(x1)
        X2.append(x2)
        ground_truths.append(ground_truth)

        if (index + 1) % mini_batch_size_dcd == 0:
            X1 = torch.stack(X1)
            X2 = torch.stack(X2)
            ground_truths = torch.LongTensor(ground_truths)
            X1 = X1.to(device)
            X2 = X2.to(device)
            ground_truths = ground_truths.to(device)

            optimizer_d.zero_grad()
            X_cat = torch.cat([encoder(X1), encoder(X2)], 1)
            y_pred = discriminator(X_cat.detach())
            loss = loss_fn(y_pred, ground_truths)
            loss.backward()
            optimizer_d.step()
            ########## loss_mean.append(loss.item())
            X1 = []
            X2 = []
            ground_truths = []
    scheduler_g_h.step()
    scheduler_d.step()
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
    print("step3----Epoch %d/%d  accuracy: %.3f best_acc: %.3f " % (epoch + 1, num_epochs3, accuracy, best_acc))
    
    
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
