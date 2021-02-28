from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from PIL import Image
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import matplotlib.image as mpimg 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import Augmentor
import transformations

os.environ['TORCH_HOME'] = "/media/goeau/DATA/villacis/tmp/torch"


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
        if self.transform is not None:
            img = self.transform(img)

        #img = np.asarray(img).transpose(-1, 0, 1) # we have to change the dimensions from width x height x channel (WHC) to channel x width x height (CWH)
        img = torch.from_numpy(np.asarray(img)) # create the image tensor
        X = img
        y = self.labels[ID]

        return X, y
img_size = 224
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(15),
        #transforms.RandomCrop((img_size, img_size)),
        #transforms.RandomResizedCrop((img_size, img_size)),
        #transforms.CenterCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transformations.TileCircle(),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.7410, 0.7141, 0.6500], [0.0808, 0.0895, 0.1141])
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.7410, 0.7141, 0.6500], [0.0808, 0.0895, 0.1141])
    ]),
    'val_photo': transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
    ]),
}

data_dir = '/home/villacis/Desktop/villacis/datasets/plantclef_superminida_17_20_split/herbarium'
#data_dir = "/media/goeau/DATA/villacis/datasets/pccomun1720_eol_split"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']} #,
learning_rate = 0.001
batch_size = 64
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 5}
          
partition = {}
partition['train'] = []
partition['val'] = []
labels = {}
print("Preprocessing datasets")
class_name_to_id = image_datasets['train'].class_to_idx
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
    labels[i[0]] = get_class_id(i[0])


for i in image_datasets['val'].imgs:
    partition['val'].append(i[0])
    labels[i[0]] = get_class_id(i[0])
    
  
#for i in image_datasets_photo.imgs:
#    partition['val_photo'].append(i[0])
#    labels[i[0]] = get_class_id(i[0])




print("Finished preprocessing datasets")


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']} 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #"cuda" if torch.cuda.is_available() else "cpu"
class_names = image_datasets['train'].classes
num_classes = len(class_names)
print(num_classes)
print(device)

train_set = Dataset(partition['train'], labels, transform = data_transforms['train'])
train_generator = data.DataLoader(train_set, **params)

val_set = Dataset(partition['val'], labels, transform = data_transforms['val'])
val_generator = data.DataLoader(val_set, **params)

dataloaders = {}
dataloaders['train'] = train_generator
dataloaders['val'] = val_generator

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']: #, 'val', 'val_photo'
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            procesados = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                procesados += len(labels)
                inputs = inputs.to(device, non_blocking = True)
                labels = labels.to(device, non_blocking = True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_corrects = running_corrects.item()
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / procesados
            epoch_acc = (running_corrects*1.0) / procesados

            print("Loss: {} || Acc: {}".format(epoch_loss, epoch_acc))
        print( time.time() - since)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
   
    return model

def ejecutar_1():
    global learning_rate
    for i in [0.0001]: #0.001
        learning_rate = i
        ## Finetunear desde imagenet
        model_ft = models.resnet50(pretrained=True)
        ## Cargar pesos
        #pretrained_weights = torch.load('mejores_pesos_viejo.pth')
        #model_ft = models.resnet50(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        #model_ft.load_state_dict(pretrained_weights)
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).

        model_ft = model_ft.to(device)
        criterion = nn.CrossEntropyLoss()


        # Observe that all parameters are being optimized
        #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        #model_ft.fc.parameters()
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate)
        # Decay LR by a factor of 0.1 every 7 epochs
        #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
        #exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, threshold=1e-4, patience=5)
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[50], gamma=0.1)



        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                               num_epochs=55)

#visualize_model(model_ft)
ejecutar_1()
