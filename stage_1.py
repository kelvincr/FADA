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
datadir = '../dataset/'
data_dir = '../dataset/herbarium'
data_dir2 = '../dataset/photo'
#data_dir2 = '/home/villacis/Desktop/villacis/datasets/todas/todo_photo'
#datadir = '/home/villacis/Desktop/villacis/datasets/plantclef_minida_cropped'
#datadir = '/home/villacis/Desktop/villacis/datasets/special_10_ind'
num_epochs1 = 60
num_epochs2 = 150
num_epochs3 = 150


use_cuda=True if torch.cuda.is_available() else False
device=torch.device('cuda:0') if use_cuda else torch.device('cpu')
print(device)

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
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transformations.TileHerb(),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
        #transforms.Normalize([0.7410, 0.7141, 0.6500], [0.0808, 0.0895, 0.1141])
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
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


print("class to idx")
class_to_idx = data.class_to_idx
print(class_to_idx)

image_datasets = {"train": data_train, 
                   "val": data_val}
                   
print(image_datasets["train"])
print(image_datasets["val"])

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