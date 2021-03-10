import os

import torch
import torch.utils.data as data2
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader,Dataset
from torchvision import models as tmodels
import torchvision
from PIL import Image

from data_loader import FADADatasetSSTaxons
import models
import data_loader
import transformations
from losses import ContrastiveLoss, SpecLoss

src = 'herbarium'  #mnist
dst = 'photo'       #svhn
img_size = 224  #32
datadir = '/../dataset/'
data_dir = '/../dataset/herbarium'
data_dir2 = '/../dataset/photo'
#data_dir2 = '/home/villacis/Desktop/villacis/datasets/todas/todo_photo'
#datadir = '/home/villacis/Desktop/villacis/datasets/plantclef_minida_cropped'
#datadir = '/home/villacis/Desktop/villacis/datasets/special_10_ind'
num_epochs1 = 40
num_epochs2 = 150
num_epochs3 = 150


use_cuda=True if torch.cuda.is_available() else False
device=torch.device('cuda:0') if use_cuda else torch.device('cpu')


classifier = models.ClassifierPro()
classifier2 = models.ClassifierPro()
#encoder = models.Encoder()
encoder = tmodels.resnet50(pretrained=True)
#encoder = tmodels.inception_v3(pretrained=True)
#encoder.aux_logits=False
encoder.fc = nn.Sequential()
discriminator = models.DCDPro()
#discriminator = models.DCDPro(input_features=128)
ssnet = models.TaxonNet(64)
discriminator_genus = models.DCDPro()
genusnet = models.TaxonNet(510)
discriminator_family = models.DCDPro()
familynet = models.TaxonNet(151)

classifier.to(device)
encoder.to(device)
#classifier2.to(device)
#encoder2.to(device)
discriminator.to(device)
ssnet.to(device)
discriminator_genus.to(device)
genusnet.to(device)
discriminator_family.to(device)
familynet.to(device)

loss_fn=torch.nn.CrossEntropyLoss()
loss_fn2 = ContrastiveLoss()  ##quitar
loss_fn3 = SpecLoss()  ##quitar
# -----------------------------------------------------------------------------
## etapa 1: entrenar g y h
print("||||| Stage 1 |||||")
optimizer=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters())+list(ssnet.parameters())+list(genusnet.parameters())+list(familynet.parameters()), lr = 0.0001)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)


#herb std-mean
#tensor([0.0808, 0.0895, 0.1141])
#tensor([0.7410, 0.7141, 0.6500])
#photo std-mean
#tensor([0.1399, 0.1464, 0.1392])
#tensor([0.2974, 0.3233, 0.2370])

data_transforms = {
    'train': transforms.Compose([
        #transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(15),
        #transforms.RandomCrop((img_size, img_size)),
        #transforms.RandomResizedCrop((img_size, img_size)),
        #transforms.CenterCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transformations.TileHerb(),
        #transformations.ScaleChange(),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
        #transforms.Normalize([0.7410, 0.7141, 0.6500], [0.0808, 0.0895, 0.1141])
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        #transforms.Resize((img_size, img_size)),
        transformations.CropField(),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
        #transforms.Normalize([0.7410, 0.7141, 0.6500], [0.0808, 0.0895, 0.1141])
        #transforms.Normalize([0.7410, 0.7141, 0.6500], [0.0808, 0.0895, 0.1141])
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val_photo': transforms.Compose([
        #transforms.Resize((img_size, img_size)),
        transformations.CropField(),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

class Dataset(data2.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels, transform=None):
        'Initialization'
        self.labels = labels
        self.transform = transform
        self.list_IDs = list_IDs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        #X = torch.load(ID)
        img = Image.open(ID) # use pillow to open a file
        img = img.convert('RGB') #convert image to RGB channel
        if self.transform is not None:
            img = self.transform(img)

        #img = np.asarray(img).transpose(-1, 0, 1) # we have to change the dimensions from width x height x channel (WHC) to channel x width x height (CWH)
        img = torch.from_numpy(np.asarray(img)) # create the image tensor
        X = img
        y = self.labels[ID]

        return X, y


train_ratio=.8
data = datasets.ImageFolder(os.path.join(data_dir), transform=data_transforms['train'])
train_size = int(train_ratio * len(data))
test_size = len(data) - train_size
data_train, data_val = torch.utils.data.random_split(data, [train_size, test_size])


image_datasets = {"train": data_train, 
                   "val": data_val}

dataloaders_1 = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=60,
                                             shuffle=True, num_workers=6)
              for x in ['train', 'val']}
partition = {}
partition['train'] = []
partition['val'] = []
partition['val_photo'] = []
labels2 = {}

base_mapping = image_datasets['train'].class_to_idx

print("Preprocessing datasets")
class_name_to_id = image_datasets['train'].class_to_idx
id_to_class_name = {v: k for k, v in class_name_to_id.items()}
def get_class_id(path):
    class_name = os.path.split(path)[0]
    class_name = os.path.split(class_name)[1]
    try:
        class_name2 = class_name_to_id[class_name]
    except:
        class_name2 = 0
        print("Error: "+str(class_name))
    return class_name2

for i in image_datasets['train'].imgs:
    partition['train'].append(i[0])
    labels2[i[0]] = get_class_id(i[0])


for i in image_datasets['val'].imgs:
    partition['val'].append(i[0])
    labels2[i[0]] = get_class_id(i[0])

print("Finished preprocessing datasets")
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)
print(num_classes)
print(device)

batch_size = 15
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 6}





image_datasets2 = {}
#image_datasets2 = {x: datasets.ImageFolder(os.path.join(data_dir2, x),
#                                          data_transforms[x])
#                  for x in ['train']} #, 'val_photo', 'val'
image_datasets2['val'] = datasets.ImageFolder('../dataset/photo',
                                          data_transforms['val'])

# hacer la regresion
partition = {}
partition['train'] = []
partition['val'] = []
partition['val_photo'] = []
labels2 = {}
#for i in image_datasets2['train'].imgs:
#    print(i[0])
#    partition['train'].append(i[0])
#    labels2[i[0]] = get_class_id(i[0])

#
for i in image_datasets2['val'].imgs:
    partition['val'].append(i[0])
    labels2[i[0]] = get_class_id(i[0])


# for i in image_datasets2['val_photo'].imgs:
#     partition['val_photo'].append(i[0])
#     labels2[i[0]] = get_class_id(i[0])

batch_size = 10
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 6}

#train_set = Dataset(partition['train'], labels2, transform = data_transforms['train'])
#train_generator = data2.DataLoader(train_set, **params)

val_set = Dataset(partition['val'], labels2, transform = data_transforms['val'])
val_generator = data2.DataLoader(val_set, **params)

# val_photo_set = Dataset(partition['val_photo'], labels2, transform = data_transforms['val_photo'])
# val_photo_generator = data2.DataLoader(val_photo_set, **params)

dataloaders_3 = {}
#dataloaders_3['train'] = train_generator
#dataloaders_3['val'] = val_generator
# dataloaders_3['val_photo'] = val_photo_generator
test_dataloader = val_generator
# test_dataloader2 = val_photo_generator

print("Finished loading data")

"""
best_acc = 0.0
for i in range(num_epochs1):
    total = 0.0
    correct = 0.0
    for data,labels in dataloaders_1['train']:
        data=data.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        pred = classifier(encoder(data))
        _, predicted = torch.max(pred.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss=loss_fn(pred,labels)
        loss.backward()
        optimizer.step()
    accuracy2 = correct / total
    scheduler.step()
    acc = 0
    total = 0.0
    correct = 0.0
    for data, labels in dataloaders_1['val']:
        with torch.no_grad():
            data=data.to(device)
            labels=labels.to(device)
            outputs=classifier(encoder(data))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #acc += (torch.max(y_test_pred,1)[1]==labels).float().mean().item()
    #accuracy=round(acc / float(len(dataloaders_1['val'])), 3)
    accuracy = correct / total
    #print('Accuracy of the network on the 10000 test images: %d %%' % (100 * accuracy))
    if(accuracy>best_acc):
        best_acc = accuracy
        torch.save(encoder.state_dict(),'result/encoder_fada_extra.pth')
        torch.save(classifier.state_dict(), 'result/classifier_fada_extra.pth')
        #classifier.save()
    print("Phase 1 ---- Epoch {} accuracy: {} / {}".format((i+1), accuracy, accuracy2))
"""
#encoder.save()
#torch.save(encoder.state_dict(),'result/encoder_base.pth')
#classifier.save()

#encoder.load()

correct = 0.0
total = 0.0
encoder.load_state_dict(torch.load('result/fsda_encoder_extra.pth'))
classifier.load_state_dict(torch.load('result/fsda_classifier_extra.pth'))
#classifier.load()

# -----------------------------------------------------------------------------
## fase 2: entrenar dcd, congelar g y h
print("||||| Stage 2 |||||")

optimizer_D = torch.optim.Adam(list(discriminator.parameters())+list(discriminator_genus.parameters())+list(discriminator_family.parameters()), lr=0.0001) #0.001

print("Carga optimizer")
#X_s,Y_s=data_loader.sample_src(os.path.join(datadir, src, 'train'), data_transforms['train'])
#X_t,Y_t=data_loader.sample_tgt(os.path.join(datadir, dst, 'train'), data_transforms['train'], n = 7)

siamese_dataset = FADADatasetSSTaxons('/home/ubuntu/dataset/herbarium',
                                       '/home/ubuntu/dataset/photo',
                                        'train',
                                        base_mapping,
                                        class_name_to_id,
                                        id_to_class_name,
                                        img_size
                                        )
dataloader_2 = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=6,
                        batch_size=10)
print("Carga datos")

"""
for epoch in range(num_epochs2):
    loss_mean = []
    cont = 0.0
    acc = 0.0
    buenos = 0.0
    confusion_matrix = torch.zeros(4, 4)
    for X1, X2, idx1, idx2, ground_truths, op1, op2, same, img0_ss, label0_ss, img1_ss, label1_ss in dataloader_2:
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

torch.save(discriminator.state_dict(),'result/discriminator_fada_extra.pth') # discriminator_fada.pth
"""
discriminator.load_state_dict(torch.load('result/discriminator_fada_extra.pth'))
discriminator_family.load_state_dict(torch.load('result/discriminator_fada_extra.pth'))
discriminator_genus.load_state_dict(torch.load('result/discriminator_fada_extra.pth'))

# -----------------------------------------------------------------------------
## fase 3: entrenar g y h, congelar dcd
print("||||| Stage 3 |||||")
#optimizer_g_h=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters()),lr=0.0001)  #0.0001
optimizer_g_h=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters())+list(ssnet.parameters())+list(genusnet.parameters())+list(familynet.parameters()),lr=0.0001)
optimizer_d=torch.optim.Adam(list(discriminator.parameters())+list(discriminator_genus.parameters())+list(discriminator_family.parameters()),lr=0.0001) #0.001
scheduler_g_h = lr_scheduler.MultiStepLR(optimizer_g_h, milestones=[15], gamma=0.1)
scheduler_d = lr_scheduler.MultiStepLR(optimizer_d, milestones=[15], gamma=0.1)

# datasets_3 = {x: datasets.ImageFolder(os.path.join(datadir, dst, x),
#                                           data_transforms['val_photo'])
#                   for x in ['val']}
# test_dataloader = torch.utils.data.DataLoader(datasets_3['val'], batch_size=10,
#                                              shuffle=True, num_workers=6)



"""
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
with torch.no_grad():
    for data, labels in test_dataloader:
        data = data.to(device)
        labels = labels.to(device)
        y_test_pred = classifier(encoder(data))
        acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()

accuracy = round(acc / float(len(test_dataloader)), 3)
if(accuracy>best_acc):
    best_acc = accuracy
print("step3----Epoch %d/%d  accuracy: %.3f best_acc: %.3f " % (0, num_epochs3, accuracy, best_acc))
"""
best_acc = 0.0
for epoch in range(num_epochs3):
    #---training g and h , DCD is frozen

    #----training g and h, dcd frozen
    cont = 0.0
    acc = 0.0
    buenos = 0.0
    loss_1 = 0.0
    loss_2 = 0.0
    encoder.train()
    classifier.train()
    discriminator.train()
    confusion_matrix = torch.zeros(4, 4)
    for X1, X2, ground_truths_y1, ground_truths_y2, ground_truths, domains1, domains2, same, img0_ss, label0_ss, img1_ss, label1_ss, family1, family2, genus1, genus2, gt_genus, gt_family in dataloader_2:
        X1=X1.to(device)
        X2=X2.to(device)
        img0_ss = img0_ss.to(device)
        img1_ss = img1_ss.to(device)
        label0_ss = label0_ss.to(device)
        label1_ss = label1_ss.to(device)
        family1 = family1.to(device)
        family2 = family2.to(device)
        genus1 = genus1.to(device)
        genus2 = genus2.to(device)
        gt_genus = gt_genus.to(device)
        gt_family = gt_family.to(device)
        dcd_labels_genus = gt_genus.view(-1)
        dcd_labels_family = gt_family.view(-1)
        ground_truths_y1=ground_truths_y1.to(device)
        ground_truths_y2 = ground_truths_y2.to(device)
        ground_truths=ground_truths.to(device)

        dcd_labels = ground_truths.view(-1)
        ground_truths_y1 = ground_truths_y1.view(-1)
        ground_truths_y1 = ground_truths_y1.view(-1)
        optimizer_g_h.zero_grad()

        encoder_X1=encoder(X1)
        encoder_X2=encoder(X2)

        X_cat=torch.cat([encoder_X1,encoder_X2],1)
        y_pred_X1=classifier(encoder_X1)
        y_pred_X2=classifier(encoder_X2)
        y_pred_dcd=discriminator(X_cat)

        y_pred_dcd_genus=discriminator_genus(X_cat)
        y_pred_dcd_family=discriminator_family(X_cat)
        pred_genus1 = genusnet(encoder_X1)
        pred_genus2 = genusnet(encoder_X2)
        pred_family1 = familynet(encoder_X1)
        pred_family2 = familynet(encoder_X2)

        pred = torch.max(y_pred_dcd,1)[1]
        for t, p in zip(ground_truths.view(-1), pred.view(-1)):
            cont += 1.0
            if(t == p):
                buenos += 1
            confusion_matrix[t.long(), p.long()] += 1

        loss_X1=loss_fn(y_pred_X1,ground_truths_y1)
        loss_X2=loss_fn(y_pred_X2,ground_truths_y2)
        loss_dcd=0.33*(loss_fn(y_pred_dcd,dcd_labels) + loss_fn(y_pred_dcd_genus,dcd_labels_genus)+ loss_fn(y_pred_dcd_family,dcd_labels_family))
        loss_genus = 0.5*(loss_fn(pred_genus1, genus1) + loss_fn(pred_genus2, genus2))
        loss_family = 0.5*(loss_fn(pred_family1, family1) + loss_fn(pred_family2, family2))
        #loss_contrastive = loss_fn2(encoder(X1),encoder(X2),same)

        pred0_ss = ssnet(encoder(img0_ss))
        pred1_ss = ssnet(encoder(img1_ss))
        loss_ss = 0.5*(loss_fn(pred1_ss, label1_ss) + loss_fn(pred0_ss, label0_ss))

        #print(loss_X1+loss_X2)
        loss_sum = loss_X1 + loss_X2 - 0.9*loss_dcd + loss_ss + 0.3*(loss_genus+loss_family)
        loss_1 += loss_sum.item()
        loss_sum.backward()
        optimizer_g_h.step()


    loss_1 /= cont
    cont = 0.0

    #----training dcd ,g and h frozen
    for X1, X2, ground_truths_y1, ground_truths_y2, ground_truths, domains1, domains2, same, img0_ss, label0_ss, img1_ss, label1_ss, family1, family2, genus1, genus2, gt_genus, gt_family in dataloader_2:
        X1 = X1.to(device)
        X2 = X2.to(device)
        same = same.to(device)
        cont+=1.0
        ground_truths = ground_truths.to(device)
        ground_truths = ground_truths.view(-1)
        gt_genus = gt_genus.to(device)
        gt_genus = gt_genus.view(-1)

        gt_family = gt_family.to(device)
        gt_family = gt_family.view(-1)


        optimizer_d.zero_grad()
        X_cat = torch.cat([encoder(X1), encoder(X2)], 1)
        y_pred = discriminator(X_cat.detach())
        y_pred_family = discriminator_family(X_cat.detach())
        y_pred_genus = discriminator_genus(X_cat.detach())
        loss = loss_fn(y_pred, ground_truths)
        loss_genus =  loss_fn(y_pred_genus, gt_genus)
        loss_family =  loss_fn(y_pred_family, gt_family)
        loss = loss + 0.5*(loss_genus + loss_family)
        loss_2 += loss.item()
        loss.backward()
        optimizer_d.step()

    loss_2 /= cont
    #testing
    encoder.eval()
    classifier.eval()
    discriminator.eval()

    with torch.no_grad():
        acc = 0
        for data, labels in test_dataloader:
            data = data.to(device)
            labels = labels.to(device)
            y_test_pred = classifier(encoder(data))
            acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()

    accuracy = round(acc / float(len(test_dataloader)), 3)
    accuracy_dcd=round(buenos / cont, 3)
    if(accuracy>best_acc or True):
        best_acc = accuracy
        nombre_e = 'result/fsda_encoder_extra_genus_family_ss' + str(epoch)+".pth"
        nombre_c = 'result/fsda_classifier_extra_genus_family_ss' + str(epoch)+".pth"
        torch.save(encoder.state_dict(),nombre_e)
        #classifier.save()
        torch.save(classifier.state_dict(),nombre_c)
    print("step30----Epoch %d/%d  accuracy: %.3f best_acc: %.3f dcd_acc: %.3f" % (epoch + 1, num_epochs3, accuracy, best_acc, accuracy_dcd))

with torch.no_grad():
    acc = 0
    for data, labels in test_dataloader:
        data = data.to(device)
        labels = labels.to(device)
        y_test_pred = classifier(encoder(data))
        acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()

accuracy = round(acc / float(len(test_dataloader)), 3)
print("Final accuracy: {}".format(accuracy))


