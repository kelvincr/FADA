from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from PIL import Image
from torch.optim import lr_scheduler
import numpy as np
import models as models
import torchvision
from torchvision import datasets, transforms
from torchvision import models as tmodels
import matplotlib.pyplot as plt
import time
import os
import copy
import matplotlib.image as mpimg
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import imghdr
import untangle
from efficientnet_pytorch import EfficientNet

import transformations
import xml.etree.ElementTree as ET

#os.environ['TORCH_HOME'] = "/media/goeau/DATA/villacis/tmp/torch"


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


data_transforms_normal = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        #p.torch_transform(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        #transforms.Resize(256),
        #transformations.CropField(),
        transforms.CenterCrop(224),
        #p.torch_transform(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_photo': transforms.Compose([
        #transforms.Resize(256),
        transforms.CenterCrop(224),
        #p.torch_transform(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

##larger images
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((900, 600)),
        transforms.CenterCrop((850, 550)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        #p.torch_transform(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((900, 600)),
        transforms.CenterCrop((850, 550)),
        #transforms.ColorJitter(),
        #transforms.RandomRotation(30),
        #p.torch_transform(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_photo': transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.Resize((900, 600)),
        transforms.CenterCrop((850, 550)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

test_transforms = transforms.Compose([
    #transforms.Resize(256),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transformations.CropField(),
    #transformations.TileCircle(),
    transforms.CenterCrop((224, 224)),
    #p.torch_transform(),
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #transforms.Normalize([0.7410, 0.7141, 0.6500], [0.0808, 0.0895, 0.1141])
    transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
])

test_transforms_grande = transforms.Compose([
    transforms.Resize((900, 600)),
    transforms.CenterCrop((850, 550)),
    #transforms.ColorJitter(),
    #transforms.RandomRotation(30),
    #p.torch_transform(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = "/home/villacis/Desktop/villacis/datasets/plantclef20_split/herbarium"
data_dir_photo = "/home/villacis/Desktop/villacis/datasets/plantclef20_split/photo"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']} #,
image_datasets_photo = datasets.ImageFolder(os.path.join(data_dir_photo),
                                          data_transforms['val_photo'])
learning_rate = 0.001
batch_size = 15
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 24}

partition = {}
partition['train'] = []
partition['val'] = []
partition['val_photo'] = []
labels = {}
print("Preprocessing datasets")
class_name_to_id = image_datasets['train'].class_to_idx
print(class_name_to_id)
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


#for i in image_datasets['val'].imgs:
#    partition['val'].append(i[0])
#    labels[i[0]] = get_class_id(i[0])


for i in image_datasets_photo.imgs:
    partition['val_photo'].append(i[0])
    labels[i[0]] = get_class_id(i[0])




print("Finished preprocessing datasets")


#dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']} #, 'val'
"""
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=106,
                                             shuffle=True, num_workers=6)
              for x in ['train']} #, 'val'
#dataloaders_photo = torch.utils.data.DataLoader(image_datasets_photo, batch_size=112,
#                                             shuffle=True, num_workers=6)
#dataloaders['val_photo'] = dataloaders_photo
"""
#dataset_sizes['val_photo'] = len(image_datasets_photo)
"""
#for i in image_datasets['train']:
#    print(i[1])
"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #"cuda" if torch.cuda.is_available() else "cpu"
class_names = image_datasets['train'].classes
num_classes = len(class_names)
print(num_classes)
print(device)

#train_set = Dataset(partition['train'], labels, transform = data_transforms['train'])
#train_generator = data.DataLoader(train_set, **params)

#val_set = Dataset(partition['val'], labels, transform = data_transforms['val'])
#val_generator = data.DataLoader(val_set, **params)

val_photo_set = Dataset(partition['val_photo'], labels, transform = data_transforms['val_photo'])
val_photo_generator = data.DataLoader(val_photo_set, **params)

dataloaders = {}
#dataloaders['train'] = train_generator
#dataloaders['val'] = val_generator
#dataloaders['val_photo'] = val_photo_generator

def load_ckp(checkpoint_fpath, model):
    checkpoint = torch.load(checkpoint_fpath)
    print(checkpoint['epoch'])
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    return model #, optimizer, checkpoint['epoch']

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = 1.0

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res
def get_rank(outputs, label):
    _, vals = outputs.sort()
    vals = vals[0]
    rank = ((vals == label).nonzero())
    rank = rank.item()
    rank = vals.size(0) - rank
    rank = rank * 1.0
    resp = 1.0 / rank
    return resp
def test_model(encoder, classifier):
    encoder.eval()
    classifier.eval()
    running_corrects = 0
    #/home/villacis/Desktop/villacis/datasets/plantclef20_src/photo
    data_dir = '/home/villacis/villacis/datasets/plantclef20_split/photo/val_peq'
    #data_dir = '/home/villacis/Desktop/villacis/datasets/plantclef20_src/photo'
    #data_dir = "/home/villacis/pc2020/photo"
    classes = os.listdir(data_dir)
    total1 = []
    total5 = []
    mrr = 0.0
    queries= 0
    obsids = {}
    obsidclase = {}
    for class_name in classes:
        source_dir = os.path.join(data_dir, class_name)
        print(class_name)
        procesados = 0
        acc1 = 0
        acc5 = 0
        for f in os.listdir(source_dir):
            if (f.endswith('.jpg') or f.endswith('.JPG')):
                procesados += 1
                base = os.path.basename(f)
                base = os.path.splitext(base)[0]
                image_name = os.path.join(source_dir, f)
                xml_base_name = "/home/villacis/villacis/datasets/plantclef20_src/photo"
                #xml_base_name = "/home/villacis/pc2020/photo"
                xml_base_name = os.path.join(xml_base_name, class_name)
                xml_name = os.path.join(xml_base_name,  base + ".xml")
                file_text = open(xml_name, 'r').read()
                file_text = file_text.replace("&", "and")
                obj = untangle.parse(file_text)
                obsid = obj.Image.ObservationId.cdata
                obsid = int(obsid)
                imagen = Image.open(image_name)
                imagen = imagen.convert("RGB")
                with torch.no_grad():
                    imagen = test_transforms(imagen)
                    imagen = imagen.to(device)
                    imagen = imagen.unsqueeze(0)
                    label = class_name_to_id[class_name]
                    label = torch.tensor(label)
                    label = label.to(device)
                    outputs = classifier(encoder(imagen))
                    if obsid in obsids:
                        obsids[obsid]+=outputs
                    else:
                        obsids[obsid] = outputs
                        obsidclase[obsid] = label.item()
                    mrr += get_rank(outputs, label.item())
                    queries += 1
                    _, preds = torch.max(outputs, 1)
                    tacc1, tacc5 = accuracy(outputs, label, topk=(1, 5))
                    acc1 += tacc1
                    acc5 += tacc5
        procesados *= 1.0
        if(procesados < 0.95):
            continue
        acc1 = (acc1*1.0)/(procesados)
        acc5 = (acc5*1.0)/(procesados)
        total1.append(acc1.item())
        total5.append(acc5.item())
        print("Clase: "+str(class_name)+" ||| acc1: "+str(acc1)+" ||| acc5: "+str(acc5))#
    total1 = torch.tensor(total1)
    total5 = torch.tensor(total5)
    total1 = torch.mean(total1)
    total5 = torch.mean(total5)
    print("Todas las clases")
    print(total1)
    print(total5)
    print("MRR fotos: ")
    print(mrr)
    print(queries)
    queries = queries*1.0
    mrr = mrr/queries
    print(mrr)
    mrr = 0.0
    queries = 0
    acc1 = 0.0
    acc5 = 0.0

    for obsid in obsids.keys():
        mrr+=get_rank(obsids[obsid], obsidclase[obsid])
        _, preds = torch.max(obsids[obsid], 1)

        label = obsidclase[obsid]
        label = torch.tensor(label)
        label = label.to(device)
        tacc1, tacc5 = accuracy(obsids[obsid], label, topk=(1, 5))
        acc1 += tacc1
        acc5 += tacc5
        queries+=1
    print("MRR obsid: ")
    print(mrr)
    print(queries)
    queries = queries*1.0
    mrr = mrr/queries
    print(mrr)
    print("Accs por topk")
    acc1 = (acc1*1.0)/(queries)
    acc5 = (acc5*1.0)/(queries)
    print(acc1)
    print(acc5)
def ejecutar_1():
    classifier = models.ClassifierPro()
    encoder = tmodels.resnet50(pretrained=True)
    encoder.fc = nn.Sequential()
    classifier.to(device)
    encoder.to(device)
    encoder.load_state_dict(torch.load('r50_encoder_taxons.pth'))
    classifier.load_state_dict(torch.load('r50_classifier_taxons.pth'))
    #encoder.load_state_dict(torch.load('result/encoder_fullfull1.pth'))
    #classifier.load_state_dict(torch.load('result/classifier_fullfull1.pth'))
    #encoder.load_state_dict(torch.load('encoder_fullfullfull.pth'))
    #classifier.load_state_dict(torch.load('classifier_fullfullfull.pth'))
    #encoder.load_state_dict(torch.load('result/encoder_full_ss.pth'))
    #classifier.load_state_dict(torch.load('result/classifier_full_ss.pth'))
    #encoder.load_state_dict(torch.load('result/encoder_fullextra.pth'))
    #classifier.load_state_dict(torch.load('result/classifier_fullextra.pth'))
    #resnet
    #model_ft = models.resnet50(pretrained=True)
        ## Cargar pesos
        #pretrained_weights = torch.load('mejores_pesos_viejo.pth')
        #model_ft = models.resnet50(pretrained=False)
    #num_ftrs = model_ft.fc.in_features
    #model_ft.fc = nn.Linear(num_ftrs, num_classes)
    #pretrained_weights = torch.load('pesos_herb_normal_resnet.pth')
    #model_ft = models.resnet50(pretrained=False)
    #model_ft.load_state_dict(pretrained_weights)
    #resnet very large
    #model_ft = models.resnet50(pretrained=True)
    ## Cargar pesos
    #pretrained_weights = torch.load('mejores_pesos_viejo.pth')
    #model_ft = models.resnet50(pretrained=False)
    #model_ft.avgpool = nn.AvgPool2d(kernel_size=(27, 18), stride=1)
    #num_ftrs = model_ft.fc.in_features
    #model_ft.fc = nn.Linear(num_ftrs, num_classes)
    #inception
    #model_ft = models.inception_v3(pretrained=True)
    #model_ft.aux_logits=False
    #num_ftrs = model_ft.fc.in_features
    #model_ft.fc = nn.Linear(num_ftrs, num_classes)
        #model_ft.load_state_dict(pretrained_weights)
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).

    #efficientnet
    #model_ft = EfficientNet.from_pretrained('efficientnet-b4', num_classes=997)

    #model_ft = load_ckp("pesos_herbinception_90.pth", model_ft)
    #model_ft = model_ft.to(device)
    #encoder = tmodels.resnet50(pretrained=True)
    test_model(encoder, classifier)


#visualize_model(model_ft)
ejecutar_1()
