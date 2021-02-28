import os

import torch
import torch.utils.data as data
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader,Dataset
from torchvision import models as models
import transformations
import torchvision
from PIL import Image

src = 'herbarium'  #mnist
dst = 'photo'       #svhn

use_cuda=True if torch.cuda.is_available() else False
device=torch.device('cuda:0') if use_cuda else torch.device('cpu')

model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 997)
model_ft.load_state_dict(torch.load('result/r50_extra.pth'))
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()

num_epochs1 = 80
learning_rate = 0.003 #0.003
img_size = 224


optimizer=torch.optim.Adam(model_ft.parameters(), lr = learning_rate)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)

def get_class_id(path):
    class_name = os.path.split(path)[0]
    class_name = os.path.split(class_name)[1]
    try:
        class_name2 = class_name_to_id[class_name]
    except:
        class_name2 = 0
        print("Error: "+str(class_name))
    return class_name2

img_size = 224
data_transforms = {
    'train': transforms.Compose([
        transformations.CropField(),
        transforms.RandomRotation(15),
        #transforms.RandomCrop((img_size, img_size)),
        #transforms.RandomResizedCrop((img_size, img_size)),
        #transforms.CenterCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        #transformations.TileCircle(),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #transforms.Normalize([0.7410, 0.7141, 0.6500], [0.0808, 0.0895, 0.1141])
        transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
    ]),
    'val': transforms.Compose([
        transformations.CropField(),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        #transforms.Normalize([0.7410, 0.7141, 0.6500], [0.0808, 0.0895, 0.1141])
        transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
    ]),
    'val_photo': transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        #transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
    ]),
}

data_dir = '/home/villacis/todo_photo'
#data_dir = '/home/villacis/todo_split/train'
#data_dir2 = '/home/villacis/todo_split/val'
data_dir = '/home/villacis/villacis/datasets/todas/todo_photo'
data_dir2 = '/home/villacis/villacis/datasets/plantclef20_split/photo/val'
data_dir3 = '/home/villacis/villacis/datasets/plantclef20_split/herbarium/train'

class Dataset(data.Dataset):
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
        
        img = self.transform(img)

        #img = np.asarray(img).transpose(-1, 0, 1) # we have to change the dimensions from width x height x channel (WHC) to channel x width x height (CWH)
        img = torch.from_numpy(np.asarray(img)) # create the image tensor
        X = img
        y = self.labels[ID]

        return X, y

# procesamiento de los datasets
print("Preprocessing datasets")
image_datasets = {}

image_datasets['viejo'] = datasets.ImageFolder(data_dir3,
                                          data_transforms['train'])

image_datasets['train'] = datasets.ImageFolder(data_dir,
                                          data_transforms['train'])

image_datasets['val'] = datasets.ImageFolder(data_dir2,
                                          data_transforms['val'])


partition = {}
partition['train'] = []
partition['val'] = []
partition['val_photo'] = []
labels2 = {}

base_mapping = image_datasets['viejo'].class_to_idx
class_name_to_id = image_datasets['viejo'].class_to_idx

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

batch_size = 40
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 6}

train_set = Dataset(partition['train'], labels2, transform = data_transforms['train'])
train_generator = DataLoader(train_set, **params)

val_set = Dataset(partition['val'], labels2, transform = data_transforms['val'])
val_generator = DataLoader(val_set, **params)

#val_photo_set = Dataset(partition['val_photo'], labels2, transform = data_transforms['val_photo'])
#val_photo_generator = data.DataLoader(val_photo_set, **params)

dataloaders_1 = {}
dataloaders_1['train'] = train_generator
dataloaders_1['val'] = val_generator
#dataloaders_1['val_photo'] = val_photo_generator
print("Finished processing datasets")
# final del procesamiento de los datasets

best_acc = 0.0
for i in range(num_epochs1):
    for data,labels in dataloaders_1['train']:
        data=data.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        pred = model_ft(data)
        loss=criterion(pred,labels)
        loss.backward()
        optimizer.step()
    scheduler.step()
    total = 0.0
    correct = 0.0
    for data, labels in dataloaders_1['val']:
        with torch.no_grad():
            data=data.to(device)
            labels=labels.to(device)
            outputs=model_ft(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    if(accuracy>best_acc):
        best_acc = accuracy
        torch.save(model_ft.state_dict(),'result/r50_extra_finetuned.pth')
        #torch.save(classifier.state_dict(), 'result/classifier_full.pth')
        #classifier.save()

    print("Phase 1 ---- Epoch {} accuracy: {}".format((i+1), accuracy))
