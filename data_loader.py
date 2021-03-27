import os
from os.path import join
import numpy as np
import random
from PIL import Image

import torch
import torch.utils.data as data
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset

import transformations
import dicts
#photo std-mean
#tensor([0.1399, 0.1464, 0.1392])
#tensor([0.2974, 0.3233, 0.2370])
class FADADataset(Dataset):

    def __init__(self,src_path, dst_path, stage, base_mapping, super_class_to_idx, image_size = 224):
        self.image_size = image_size
        self.base_mapping = base_mapping
        self.transform_src = transforms.Compose([
                                            transforms.RandomRotation(15),
                                            #transforms.RandomCrop((img_size, img_size)),
                                            #transforms.RandomResizedCrop((img_size, img_size)),
                                            #transforms.CenterCrop((img_size, img_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ColorJitter(),
                                            #transformations.TileHerb(),
                                            transformations.TileCircle(),
                                            transformations.ScaleChange(),
                                            transforms.CenterCrop((image_size, image_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.7410, 0.7141, 0.6500], [0.0808, 0.0895, 0.1141])
                                            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            #transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
                                            ])
        self.transform_dst = transforms.Compose([
                                            #transforms.RandomRotation(15),
                                            #transforms.RandomHorizontalFlip(),
                                            #transforms.ColorJitter(),

                                            transformations.CropField(),
                                            transforms.CenterCrop((image_size, image_size)),
                                            #transforms.RandomCrop((224, 224)),
                                            transforms.ToTensor(),
                                            #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
                                            ])
        self.dataset_src = datasets.ImageFolder(src_path, self.transform_src)
        self.dataset_dst = datasets.ImageFolder(dst_path, self.transform_dst) #os.path.join(dst_path, stage)
        self.idx_imgs_src = {}
        self.idx_imgs_dst = {}
        #self.class_to_idx = self.dataset_src.class_to_idx
        self.class_to_idx = super_class_to_idx
        self.classes = []
        self.mini_classes = []
        self.idx_to_class_dst = {v: k for k, v in self.dataset_dst.class_to_idx.items()}
        self.idx_to_class_src = {v: k for k, v in self.dataset_src.class_to_idx.items()}
        for i2 in self.dataset_src.class_to_idx:
            i = self.base_mapping[i2]
            self.idx_imgs_src[i] = []
            self.idx_imgs_dst[i] = []
            self.classes.append(i)
        for i2 in self.dataset_dst.class_to_idx:
            i = self.base_mapping[i2]
            self.mini_classes.append(i)
        for i in self.dataset_src.imgs:
            idx_base = self.base_mapping[self.idx_to_class_src[i[1]]]
            self.idx_imgs_src[idx_base].append(i[0])
        for i in self.dataset_dst.imgs:
            idx_base = self.base_mapping[self.idx_to_class_dst[i[1]]]
            self.idx_imgs_dst[idx_base].append(i[0])
    def __getitem__(self,index):

        dist = random.randint(1, 100)
        if(dist<6):
            opcion=1
        elif(dist<51):
            opcion = 2
        elif(dist<56):
            opcion = 3
        else:
            opcion = 4
        opcion = random.randint(1, 4)
        #opcion = random.choice([2, 4])
        label = torch.tensor([opcion])
        #0 = s, 1 = t
        op1 = 0
        op2 = 0
        same = 0
        if(opcion == 1):
            #sx, sx
            idx = random.choice(self.classes)
            idx1 = idx
            idx2 = idx
            img0_path = random.choice(self.idx_imgs_src[idx])
            img1_path = random.choice(self.idx_imgs_src[idx])
        if(opcion == 2):
            #tx, sx
            idx = random.choice(self.mini_classes)
            idx1 = idx
            idx2 = idx
            img0_path = random.choice(self.idx_imgs_src[idx])
            img1_path = random.choice(self.idx_imgs_dst[idx])
            op1 = 1
        if(opcion == 3):
            #sx, sy
            idx1 = random.choice(self.classes)
            idx2 = idx1
            while(idx2 == idx1):
                idx2 = random.choice(self.classes)
            img0_path = random.choice(self.idx_imgs_src[idx1])
            img1_path = random.choice(self.idx_imgs_src[idx2])
            same = 1
        if(opcion == 4):
            #sx, ty
            idx2 = random.choice(self.mini_classes)
            idx1 = idx2
            while(idx1 == idx2):
                idx1 = random.choice(self.classes)
            img0_path = random.choice(self.idx_imgs_src[idx1])
            img1_path = random.choice(self.idx_imgs_dst[idx2])
            op2 = 1
            same = 1

        img0 = Image.open(img0_path)
        img1 = Image.open(img1_path)
        img0 = img0.convert("RGB")
        img1 = img1.convert("RGB")

        #idx1 = torch.tensor([idx1])
        #idx2 = torch.tensor([idx2])
        if(op1 == 0):
            img0 = self.transform_src(img0)
        if(op1 == 1):
            img0 = self.transform_dst(img0)
        if(op2 == 0):
            img1 = self.transform_src(img1)
        if(op2 == 1):
            img1 = self.transform_dst(img1)

        #label/=2
        label -= 1
        return img0, img1, idx1, idx2, label, op1, op2, same

    def __len__(self):
        return 2*min(len(self.dataset_src), len(self.dataset_dst))
        #return 4*min(len(self.dataset_src), len(self.dataset_dst))
        #return 120


class FADADatasetSSTaxons(Dataset):

    def __init__(self,src_path, dst_path, stage, base_mapping, super_class_to_idx, id_to_class_name, image_size = 224):
        self.image_size = image_size
        self.base_mapping = base_mapping
        self.transform_src = transforms.Compose([
                                            transforms.RandomRotation(15),
                                            #transforms.RandomCrop((img_size, img_size)),
                                            #transforms.RandomResizedCrop((img_size, img_size)),
                                            #transforms.CenterCrop((img_size, img_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ColorJitter(),
                                            transformations.TileHerb(),
                                            #transformations.TileCircle(),
                                            transformations.ScaleChange(),
                                            transforms.CenterCrop((image_size, image_size)),

                                            ])
        self.transform_src2 = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.7410, 0.7141, 0.6500], [0.0808, 0.0895, 0.1141])
                                            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            #transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
                                            ])
        self.transform_dst = transforms.Compose([
                                            transforms.RandomRotation(15),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ColorJitter(),

                                            transformations.CropField(),
                                            transforms.CenterCrop((image_size, image_size)),
                                            #transforms.RandomCrop((224, 224)),

                                            ])
        self.transform_dst2 = transforms.Compose([
                                            transforms.ToTensor(),
                                            #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
                                            ])
        self.transform_ball = transforms.Compose([
                                            #transformations.AddRandomRotation()
                                            #transformations.AddBalldo()
                                            transformations.AddJigsaw()
                                            ])
        #
        self.dataset_src = datasets.ImageFolder(src_path, self.transform_src)
        #self.dataset_dst = datasets.ImageFolder(os.path.join(dst_path, stage), self.transform_dst)
        self.dataset_dst = datasets.ImageFolder(dst_path, self.transform_dst)
        self.idx_imgs_src = {}
        self.idx_imgs_dst = {}
        #self.class_to_idx = self.dataset_src.class_to_idx
        self.id_to_class_name = id_to_class_name
        self.class_to_idx = super_class_to_idx
        self.classes = []
        self.mini_classes = []
        self.idx_to_class_dst = {v: k for k, v in self.dataset_dst.class_to_idx.items()}
        self.idx_to_class_src = {v: k for k, v in self.dataset_src.class_to_idx.items()}
        for i2 in self.dataset_src.class_to_idx:
            i = self.base_mapping[i2]
            self.idx_imgs_src[i] = []
            self.idx_imgs_dst[i] = []
            self.classes.append(i)
        for i2 in self.dataset_dst.class_to_idx:
            i = self.base_mapping[i2]
            self.mini_classes.append(i)
        for i in self.dataset_src.imgs:
            idx_base = self.base_mapping[self.idx_to_class_src[i[1]]]
            self.idx_imgs_src[idx_base].append(i[0])
        for i in self.dataset_dst.imgs:
            idx_base = self.base_mapping[self.idx_to_class_dst[i[1]]]
            self.idx_imgs_dst[idx_base].append(i[0])
    def __getitem__(self,index):

        dist = random.randint(1, 100)
        if(dist<6):
            opcion=1
        elif(dist<51):
            opcion = 2
        elif(dist<56):
            opcion = 3
        else:
            opcion = 4
        opcion = random.randint(1, 4)
        #opcion = random.choice([2, 4])
        label = torch.tensor([opcion])
        label_genus = torch.tensor([opcion])
        label_family = torch.tensor([opcion])
        #0 = s, 1 = t
        op1 = 0
        op2 = 0
        same = 0
        if(opcion == 1):
            #sx, sx
            idx = random.choice(self.classes)
            idx1 = idx
            idx2 = idx
            img0_path = random.choice(self.idx_imgs_src[idx])
            img1_path = random.choice(self.idx_imgs_src[idx])
        if(opcion == 2):
            #tx, sx
            idx = random.choice(self.mini_classes)
            idx1 = idx
            idx2 = idx
            img0_path = random.choice(self.idx_imgs_src[idx])
            img1_path = random.choice(self.idx_imgs_dst[idx])
            op1 = 1
        if(opcion == 3):
            #sx, sy
            idx1 = random.choice(self.classes)
            idx2 = idx1
            while(idx2 == idx1):
                idx2 = random.choice(self.classes)
            img0_path = random.choice(self.idx_imgs_src[idx1])
            img1_path = random.choice(self.idx_imgs_src[idx2])
            same = 1
        if(opcion == 4):
            #sx, ty
            idx2 = random.choice(self.mini_classes)
            idx1 = idx2
            while(idx1 == idx2):
                idx1 = random.choice(self.classes)
            img0_path = random.choice(self.idx_imgs_src[idx1])
            img1_path = random.choice(self.idx_imgs_dst[idx2])
            op2 = 1
            same = 1

        img0 = Image.open(img0_path)
        img0_ss = Image.open(img0_path)
        img1 = Image.open(img1_path)
        img1_ss = Image.open(img1_path)
        img0 = img0.convert("RGB")
        img0_ss = img0_ss.convert("RGB")
        img1 = img1.convert("RGB")
        img1_ss = img1_ss.convert("RGB")

        #idx1 = torch.tensor([idx1])
        #idx2 = torch.tensor([idx2])
        labelss_0 = 1
        labelss_1 = 1
        if(op1 == 0):
            img0 = self.transform_src(img0)
            img0_ss = self.transform_src(img0_ss)
            img0 = self.transform_src2(img0)
            img0_ss = self.transform_ball(img0_ss)
            labelss_0 = img0_ss[1]
            img0_ss = img0_ss[0]
            img0_ss = self.transform_src2(img0_ss)
        if(op1 == 1):
            img0 = self.transform_dst(img0)
            img0_ss = self.transform_dst(img0_ss)
            img0 = self.transform_dst2(img0)
            img0_ss = self.transform_ball(img0_ss)
            labelss_0 = img0_ss[1]
            img0_ss = img0_ss[0]
            img0_ss = self.transform_dst2(img0_ss)
        if(op2 == 0):
            img1 = self.transform_src(img1)
            img1_ss = self.transform_src(img1_ss)
            img1 = self.transform_src2(img1)
            img1_ss = self.transform_ball(img1_ss)
            labelss_1 = img1_ss[1]
            img1_ss = img1_ss[0]
            img1_ss = self.transform_src2(img1_ss)
        if(op2 == 1):
            img1 = self.transform_dst(img1)
            img1_ss = self.transform_dst(img1_ss)
            img1 = self.transform_dst2(img1)
            img1_ss = self.transform_ball(img1_ss)
            labelss_1 = img1_ss[1]
            img1_ss = img1_ss[0]
            img1_ss = self.transform_dst2(img1_ss)


        clase1 = self.id_to_class_name[idx1]
        family1 = dicts.family[clase1]
        genus1 = dicts.genus[clase1]

        clase2 = self.id_to_class_name[idx2]
        family2 = dicts.family[clase2]
        genus2 = dicts.genus[clase2]

        if(opcion==3):
            #label genus: same domain, different genus
            if(genus1==genus2):
                label_genus = torch.tensor([1])
            if(family1==family2):
                label_family = torch.tensor([1])
        if(opcion==4):
            #label genus: different domain, different genus
            if(genus1==genus2):
                label_genus = torch.tensor([2])
            if(family1==family2):
                label_family = torch.tensor([2])


        #label/=2
        label -= 1
        labelss_0-=1
        labelss_1-=1
        label_genus-=1
        label_family-=1
        return img0, img1, idx1, idx2, label, op1, op2, same, img0_ss, labelss_0, img1_ss, labelss_1, family1, family2, genus1, genus2, label_genus, label_family

    def __len__(self):
        return 2*min(len(self.dataset_src), len(self.dataset_dst))
        #return 4*min(len(self.dataset_src), len(self.dataset_dst))
        #return 120


class FADADatasetTaxons(Dataset):

    def __init__(self,src_path, dst_path, stage, base_mapping, super_class_to_idx, id_to_class_name, image_size = 224):
        self.image_size = image_size
        self.base_mapping = base_mapping
        self.id_to_class_name = id_to_class_name
        self.transform_src = transforms.Compose([
                                            transforms.RandomRotation(15),
                                            #transforms.RandomCrop((img_size, img_size)),
                                            #transforms.RandomResizedCrop((img_size, img_size)),
                                            #transforms.CenterCrop((img_size, img_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ColorJitter(),
                                            #transformations.TileHerb(),
                                            transformations.TileCircle(),
                                            transformations.ScaleChange(),
                                            transforms.CenterCrop((image_size, image_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.7410, 0.7141, 0.6500], [0.0808, 0.0895, 0.1141])
                                            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            #transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
                                            ])
        self.transform_dst = transforms.Compose([
                                            #transforms.RandomRotation(15),
                                            #transforms.RandomHorizontalFlip(),
                                            #transforms.ColorJitter(),

                                            transformations.CropField(),
                                            transforms.CenterCrop((image_size, image_size)),
                                            #transforms.RandomCrop((224, 224)),
                                            transforms.ToTensor(),
                                            #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
                                            ])
        self.dataset_src = datasets.ImageFolder(os.path.join(src_path, stage), self.transform_src)
        self.dataset_dst = datasets.ImageFolder(os.path.join(dst_path, stage), self.transform_dst)
        self.idx_imgs_src = {}
        self.idx_imgs_dst = {}
        #self.class_to_idx = self.dataset_src.class_to_idx
        self.class_to_idx = super_class_to_idx
        self.classes = []
        self.mini_classes = []
        self.idx_to_class_dst = {v: k for k, v in self.dataset_dst.class_to_idx.items()}
        self.idx_to_class_src = {v: k for k, v in self.dataset_src.class_to_idx.items()}
        for i2 in self.dataset_src.class_to_idx:
            i = self.base_mapping[i2]
            self.idx_imgs_src[i] = []
            self.idx_imgs_dst[i] = []
            self.classes.append(i)
        for i2 in self.dataset_dst.class_to_idx:
            i = self.base_mapping[i2]
            self.mini_classes.append(i)
        for i in self.dataset_src.imgs:
            idx_base = self.base_mapping[self.idx_to_class_src[i[1]]]
            self.idx_imgs_src[idx_base].append(i[0])
        for i in self.dataset_dst.imgs:
            idx_base = self.base_mapping[self.idx_to_class_dst[i[1]]]
            self.idx_imgs_dst[idx_base].append(i[0])
    def __getitem__(self,index):

        dist = random.randint(1, 100)
        if(dist<6):
            opcion=1
        elif(dist<51):
            opcion = 2
        elif(dist<56):
            opcion = 3
        else:
            opcion = 4
        opcion = random.randint(1, 4)
        #opcion = random.choice([2, 4])
        label = torch.tensor([opcion])
        label_genus = torch.tensor([opcion])
        label_family = torch.tensor([opcion])
        #0 = s, 1 = t
        op1 = 0
        op2 = 0
        same = 0
        if(opcion == 1):
            #sx, sx
            idx = random.choice(self.classes)
            idx1 = idx
            idx2 = idx
            img0_path = random.choice(self.idx_imgs_src[idx])
            img1_path = random.choice(self.idx_imgs_src[idx])
        if(opcion == 2):
            #tx, sx
            idx = random.choice(self.mini_classes)
            idx1 = idx
            idx2 = idx
            img0_path = random.choice(self.idx_imgs_src[idx])
            img1_path = random.choice(self.idx_imgs_dst[idx])
            op1 = 1
        if(opcion == 3):
            #sx, sy
            idx1 = random.choice(self.classes)
            idx2 = idx1
            while(idx2 == idx1):
                idx2 = random.choice(self.classes)
            img0_path = random.choice(self.idx_imgs_src[idx1])
            img1_path = random.choice(self.idx_imgs_src[idx2])
            same = 1
        if(opcion == 4):
            #sx, ty
            idx2 = random.choice(self.mini_classes)
            idx1 = idx2
            while(idx1 == idx2):
                idx1 = random.choice(self.classes)
            img0_path = random.choice(self.idx_imgs_src[idx1])
            img1_path = random.choice(self.idx_imgs_dst[idx2])
            op2 = 1
            same = 1

        img0 = Image.open(img0_path)
        img1 = Image.open(img1_path)
        img0 = img0.convert("RGB")
        img1 = img1.convert("RGB")

        #idx1 = torch.tensor([idx1])
        #idx2 = torch.tensor([idx2])
        if(op1 == 0):
            img0 = self.transform_src(img0)
        if(op1 == 1):
            img0 = self.transform_dst(img0)
        if(op2 == 0):
            img1 = self.transform_src(img1)
        if(op2 == 1):
            img1 = self.transform_dst(img1)

        #label/=2
        label -= 1

        clase1 = self.id_to_class_name[idx1]
        family1 = dicts.family[clase1]
        genus1 = dicts.genus[clase1]

        clase2 = self.id_to_class_name[idx2]
        family2 = dicts.family[clase2]
        genus2 = dicts.genus[clase2]

        return img0, img1, idx1, idx2, label, op1, op2, same, genus1, genus2

    def __len__(self):
        return 2*min(len(self.dataset_src), len(self.dataset_dst))
        #return 4*min(len(self.dataset_src), len(self.dataset_dst))
        #return 120

#
class FADADatasetTaxonsDCD(Dataset):

    def __init__(self,src_path, dst_path, stage, base_mapping, super_class_to_idx, id_to_class_name, image_size = 224):
        self.image_size = image_size
        self.base_mapping = base_mapping
        self.id_to_class_name = id_to_class_name
        self.transform_src = transforms.Compose([
                                            transforms.RandomRotation(15),
                                            #transforms.RandomCrop((img_size, img_size)),
                                            #transforms.RandomResizedCrop((img_size, img_size)),
                                            #transforms.CenterCrop((img_size, img_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ColorJitter(),
                                            #transformations.TileHerb(),
                                            transformations.TileCircle(),
                                            transformations.ScaleChange(),
                                            transforms.CenterCrop((image_size, image_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.7410, 0.7141, 0.6500], [0.0808, 0.0895, 0.1141])
                                            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            #transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
                                            ])
        self.transform_dst = transforms.Compose([
                                            #transforms.RandomRotation(15),
                                            #transforms.RandomHorizontalFlip(),
                                            #transforms.ColorJitter(),

                                            transformations.CropField(),
                                            transforms.CenterCrop((image_size, image_size)),
                                            #transforms.RandomCrop((224, 224)),
                                            transforms.ToTensor(),
                                            #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
                                            ])
        self.dataset_src = datasets.ImageFolder(os.path.join(src_path, stage), self.transform_src)
        self.dataset_dst = datasets.ImageFolder(os.path.join(dst_path, stage), self.transform_dst)
        self.idx_imgs_src = {}
        self.idx_imgs_dst = {}
        #self.class_to_idx = self.dataset_src.class_to_idx
        self.class_to_idx = super_class_to_idx
        self.classes = []
        self.mini_classes = []
        self.idx_to_class_dst = {v: k for k, v in self.dataset_dst.class_to_idx.items()}
        self.idx_to_class_src = {v: k for k, v in self.dataset_src.class_to_idx.items()}
        for i2 in self.dataset_src.class_to_idx:
            i = self.base_mapping[i2]
            self.idx_imgs_src[i] = []
            self.idx_imgs_dst[i] = []
            self.classes.append(i)
        for i2 in self.dataset_dst.class_to_idx:
            i = self.base_mapping[i2]
            self.mini_classes.append(i)
        for i in self.dataset_src.imgs:
            idx_base = self.base_mapping[self.idx_to_class_src[i[1]]]
            self.idx_imgs_src[idx_base].append(i[0])
        for i in self.dataset_dst.imgs:
            idx_base = self.base_mapping[self.idx_to_class_dst[i[1]]]
            self.idx_imgs_dst[idx_base].append(i[0])
    def __getitem__(self,index):

        dist = random.randint(1, 100)
        if(dist<6):
            opcion=1
        elif(dist<51):
            opcion = 2
        elif(dist<56):
            opcion = 3
        else:
            opcion = 4
        opcion = random.randint(1, 4)
        #opcion = random.choice([2, 4])
        label = torch.tensor([opcion])
        label_genus = torch.tensor([opcion])
        label_family = torch.tensor([opcion])
        #0 = s, 1 = t
        op1 = 0
        op2 = 0
        same = 0
        if(opcion == 1):
            #sx, sx
            #label_genus se mantiene igual, same genus, same domain
            idx = random.choice(self.classes)
            idx1 = idx
            idx2 = idx
            img0_path = random.choice(self.idx_imgs_src[idx])
            img1_path = random.choice(self.idx_imgs_src[idx])
        if(opcion == 2):
            #tx, sx
            #label genus: same genus, different domain
            idx = random.choice(self.mini_classes)
            idx1 = idx
            idx2 = idx
            img0_path = random.choice(self.idx_imgs_src[idx])
            img1_path = random.choice(self.idx_imgs_dst[idx])
            op1 = 1
        if(opcion == 3):
            #sx, sy
            #label genus: same domain, different genus
            idx1 = random.choice(self.classes)
            idx2 = idx1
            while(idx2 == idx1):
                idx2 = random.choice(self.classes)
            img0_path = random.choice(self.idx_imgs_src[idx1])
            img1_path = random.choice(self.idx_imgs_src[idx2])
            same = 1
        if(opcion == 4):
            #sx, ty
            #label genus: different domain, different genus
            idx2 = random.choice(self.mini_classes)
            idx1 = idx2
            while(idx1 == idx2):
                idx1 = random.choice(self.classes)
            img0_path = random.choice(self.idx_imgs_src[idx1])
            img1_path = random.choice(self.idx_imgs_dst[idx2])
            op2 = 1
            same = 1

        img0 = Image.open(img0_path)
        img1 = Image.open(img1_path)
        img0 = img0.convert("RGB")
        img1 = img1.convert("RGB")

        #idx1 = torch.tensor([idx1])
        #idx2 = torch.tensor([idx2])
        if(op1 == 0):
            img0 = self.transform_src(img0)
        if(op1 == 1):
            img0 = self.transform_dst(img0)
        if(op2 == 0):
            img1 = self.transform_src(img1)
        if(op2 == 1):
            img1 = self.transform_dst(img1)


        clase1 = self.id_to_class_name[idx1]
        family1 = dicts.family[clase1]
        genus1 = dicts.genus[clase1]

        clase2 = self.id_to_class_name[idx2]
        family2 = dicts.family[clase2]
        genus2 = dicts.genus[clase2]

        if(opcion==3):
            #label genus: same domain, different genus
            if(genus1==genus2):
                label_genus = torch.tensor([1])
            if(family1==family2):
                label_family = torch.tensor([1])
        if(opcion==4):
            #label genus: different domain, different genus
            if(genus1==genus2):
                label_genus = torch.tensor([2])
            if(family1==family2):
                label_family = torch.tensor([2])
        label -= 1
        label_genus -= 1
        label_family -= 1

        return img0, img1, idx1, idx2, label, op1, op2, same, genus1, genus2, label_genus, label_family

    def __len__(self):
        return 2*min(len(self.dataset_src), len(self.dataset_dst))
        #return 4*min(len(self.dataset_src), len(self.dataset_dst))
        #return 120




class FADADatasetSS(Dataset):

    def __init__(self,src_path, dst_path, stage, base_mapping, super_class_to_idx, image_size = 224):
        self.image_size = image_size
        self.base_mapping = base_mapping
        self.transform_src = transforms.Compose([
                                            transforms.RandomRotation(15),
                                            #transforms.RandomCrop((img_size, img_size)),
                                            #transforms.RandomResizedCrop((img_size, img_size)),
                                            #transforms.CenterCrop((img_size, img_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ColorJitter(),
                                            transformations.TileHerb(),
                                            #transformations.TileCircle(),
                                            transformations.ScaleChange(),
                                            transforms.CenterCrop((image_size, image_size)),

                                            ])
        self.transform_src2 = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.7410, 0.7141, 0.6500], [0.0808, 0.0895, 0.1141])
                                            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            #transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
                                            ])
        self.transform_dst = transforms.Compose([
                                            transforms.RandomRotation(15),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ColorJitter(),

                                            transformations.CropField(),
                                            transforms.CenterCrop((image_size, image_size)),
                                            #transforms.RandomCrop((224, 224)),

                                            ])
        self.transform_dst2 = transforms.Compose([
                                            transforms.ToTensor(),
                                            #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
                                            ])
        self.transform_ball = transforms.Compose([
                                            #transformations.AddRandomRotation()
                                            #transformations.AddBalldo()
                                            transformations.AddJigsaw()
                                            ])
        self.dataset_src = datasets.ImageFolder(os.path.join(src_path, stage), self.transform_src)
        #self.dataset_dst = datasets.ImageFolder(os.path.join(dst_path, stage), self.transform_dst)
        self.dataset_dst = datasets.ImageFolder(dst_path, self.transform_dst)
        self.idx_imgs_src = {}
        self.idx_imgs_dst = {}
        #self.class_to_idx = self.dataset_src.class_to_idx
        self.class_to_idx = super_class_to_idx
        self.classes = []
        self.mini_classes = []
        self.idx_to_class_dst = {v: k for k, v in self.dataset_dst.class_to_idx.items()}
        self.idx_to_class_src = {v: k for k, v in self.dataset_src.class_to_idx.items()}
        for i2 in self.dataset_src.class_to_idx:
            i = self.base_mapping[i2]
            self.idx_imgs_src[i] = []
            self.idx_imgs_dst[i] = []
            self.classes.append(i)
        for i2 in self.dataset_dst.class_to_idx:
            i = self.base_mapping[i2]
            self.mini_classes.append(i)
        for i in self.dataset_src.imgs:
            idx_base = self.base_mapping[self.idx_to_class_src[i[1]]]
            self.idx_imgs_src[idx_base].append(i[0])
        for i in self.dataset_dst.imgs:
            idx_base = self.base_mapping[self.idx_to_class_dst[i[1]]]
            self.idx_imgs_dst[idx_base].append(i[0])
    def __getitem__(self,index):

        dist = random.randint(1, 100)
        if(dist<6):
            opcion=1
        elif(dist<51):
            opcion = 2
        elif(dist<56):
            opcion = 3
        else:
            opcion = 4
        opcion = random.randint(1, 4)
        #opcion = random.choice([2, 4])
        label = torch.tensor([opcion])
        #0 = s, 1 = t
        op1 = 0
        op2 = 0
        same = 0
        if(opcion == 1):
            #sx, sx
            idx = random.choice(self.classes)
            while(len(self.idx_imgs_src[idx]) == 0):
                idx = random.choice(self.classes)
            idx1 = idx
            idx2 = idx
            img0_path = random.choice(self.idx_imgs_src[idx])
            img1_path = random.choice(self.idx_imgs_src[idx])
        if(opcion == 2):
            #tx, sx
            idx = random.choice(self.mini_classes)
            while(len(self.idx_imgs_src[idx]) == 0):
                idx = random.choice(self.mini_classes)
            idx1 = idx
            idx2 = idx
            img0_path = random.choice(self.idx_imgs_src[idx])
            img1_path = random.choice(self.idx_imgs_dst[idx])
            op1 = 1
        if(opcion == 3):
            #sx, sy
            idx1 = random.choice(self.classes)
            while(len(self.idx_imgs_src[idx1]) == 0):
                idx1 = random.choice(self.classes)
            idx2 = idx1
            while(idx2 == idx1 and len(self.idx_imgs_src[idx2]) == 0):
                idx2 = random.choice(self.classes)
            img0_path = random.choice(self.idx_imgs_src[idx1])
            img1_path = random.choice(self.idx_imgs_src[idx2])
            same = 1
        if(opcion == 4):
            #sx, ty
            idx2 = random.choice(self.mini_classes)
            while(len(self.idx_imgs_src[idx2]) == 0):
                idx2 = random.choice(self.mini_classes)
            idx1 = idx2
            while(idx1 == idx2 and len(self.idx_imgs_src[idx1]) == 0):
                idx1 = random.choice(self.classes)
            img0_path = random.choice(self.idx_imgs_src[idx1])
            img1_path = random.choice(self.idx_imgs_dst[idx2])
            op2 = 1
            same = 1

        img0 = Image.open(img0_path)
        img0_ss = Image.open(img0_path)
        img1 = Image.open(img1_path)
        img1_ss = Image.open(img1_path)
        img0 = img0.convert("RGB")
        img0_ss = img0_ss.convert("RGB")
        img1 = img1.convert("RGB")
        img1_ss = img1_ss.convert("RGB")

        #idx1 = torch.tensor([idx1])
        #idx2 = torch.tensor([idx2])
        labelss_0 = 1
        labelss_1 = 1
        if(op1 == 0):
            img0 = self.transform_src(img0)
            img0_ss = self.transform_src(img0_ss)
            img0 = self.transform_src2(img0)
            img0_ss = self.transform_ball(img0_ss)
            labelss_0 = img0_ss[1]
            img0_ss = img0_ss[0]
            img0_ss = self.transform_src2(img0_ss)
        if(op1 == 1):
            img0 = self.transform_dst(img0)
            img0_ss = self.transform_dst(img0_ss)
            img0 = self.transform_dst2(img0)
            img0_ss = self.transform_ball(img0_ss)
            labelss_0 = img0_ss[1]
            img0_ss = img0_ss[0]
            img0_ss = self.transform_dst2(img0_ss)
        if(op2 == 0):
            img1 = self.transform_src(img1)
            img1_ss = self.transform_src(img1_ss)
            img1 = self.transform_src2(img1)
            img1_ss = self.transform_ball(img1_ss)
            labelss_1 = img1_ss[1]
            img1_ss = img1_ss[0]
            img1_ss = self.transform_src2(img1_ss)
        if(op2 == 1):
            img1 = self.transform_dst(img1)
            img1_ss = self.transform_dst(img1_ss)
            img1 = self.transform_dst2(img1)
            img1_ss = self.transform_ball(img1_ss)
            labelss_1 = img1_ss[1]
            img1_ss = img1_ss[0]
            img1_ss = self.transform_dst2(img1_ss)

        #label/=2
        label -= 1
        labelss_0-=1
        labelss_1-=1
        return img0, img1, idx1, idx2, label, op1, op2, same, img0_ss, labelss_0, img1_ss, labelss_1

    def __len__(self):
        return 2*min(len(self.dataset_src), len(self.dataset_dst))
        #return 4*min(len(self.dataset_src), len(self.dataset_dst))
        #return 120

"""
To take advantage of individual pairs between herbariums and photos
"""
class FADADatasetIndividual(Dataset):

    def __init__(self,src_path, dst_path, src_path_ind, dst_path_ind, stage, base_mapping, image_size = 224):
        self.image_size = image_size
        self.base_mapping = base_mapping
        self.transform_src = transforms.Compose([
                                            transforms.RandomRotation(15),
                                            #transforms.RandomCrop((img_size, img_size)),
                                            #transforms.RandomResizedCrop((img_size, img_size)),
                                            #transforms.CenterCrop((img_size, img_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ColorJitter(),
                                            #transformations.TileHerb(),
                                            transformations.TileCircle(),
                                            transformations.ScaleChange(),
                                            transforms.CenterCrop((image_size, image_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.7410, 0.7141, 0.6500], [0.0808, 0.0895, 0.1141])
                                            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            #transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
                                            ])
        self.transform_dst = transforms.Compose([
                                            #transforms.RandomRotation(15),
                                            #transforms.RandomHorizontalFlip(),
                                            #transforms.ColorJitter(),

                                            transformations.CropField(),
                                            transforms.CenterCrop((image_size, image_size)),
                                            #transforms.RandomCrop((224, 224)),
                                            transforms.ToTensor(),
                                            #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            transforms.Normalize([0.2974, 0.3233, 0.2370], [0.1399, 0.1464, 0.1392])
                                            ])
        self.dataset_src = datasets.ImageFolder(os.path.join(src_path, stage), self.transform_src)
        self.dataset_dst = datasets.ImageFolder(os.path.join(dst_path, stage), self.transform_dst)
        self.dataset_dst_ind = datasets.ImageFolder(dst_path_ind, self.transform_dst)
        self.dataset_src_ind = datasets.ImageFolder(src_path_ind, self.transform_dst)
        self.idx_imgs_src = {}
        self.idx_imgs_dst = {}
        self.idx_imgs_src_ind = {}
        self.idx_imgs_dst_ind = {}
        self.class_to_idx = self.dataset_src.class_to_idx
        self.classes = []
        self.mini_classes = []
        self.classes_ind = []
        self.idx_to_class_dst = {v: k for k, v in self.dataset_dst.class_to_idx.items()}
        self.idx_to_class_src = {v: k for k, v in self.dataset_src.class_to_idx.items()}
        self.idx_to_class_dst_ind = {v: k for k, v in self.dataset_dst_ind.class_to_idx.items()}
        self.idx_to_class_src_ind = {v: k for k, v in self.dataset_src_ind.class_to_idx.items()}
        for i2 in base_mapping:
            i = base_mapping[i2]
            self.idx_imgs_src[i] = []
            self.idx_imgs_dst[i] = []
            self.idx_imgs_src_ind[i] = []
            self.idx_imgs_dst_ind[i] = []
        for i2 in self.dataset_src.class_to_idx:
            i = self.base_mapping[i2]
            self.idx_imgs_src[i] = []
            self.idx_imgs_dst[i] = []
            self.idx_imgs_src_ind[i] = []
            self.idx_imgs_dst_ind[i] = []
            self.classes.append(i)
        for i2 in self.dataset_dst.class_to_idx:
            i = self.base_mapping[i2]
            self.mini_classes.append(i)
        for i2 in self.dataset_dst_ind.class_to_idx:
            i = self.base_mapping[i2]
            self.classes_ind.append(i)
        print(self.classes)
        print(self.mini_classes)
        print(self.classes_ind)
        for i in self.dataset_src.imgs:
            idx_base = self.base_mapping[self.idx_to_class_src[i[1]]]
            self.idx_imgs_src[idx_base].append(i[0])
        for i in self.dataset_dst.imgs:
            idx_base = self.base_mapping[self.idx_to_class_dst[i[1]]]
            self.idx_imgs_dst[idx_base].append(i[0])

        for i in self.dataset_src_ind.imgs:
            idx_base = self.base_mapping[self.idx_to_class_src_ind[i[1]]]
            self.idx_imgs_src_ind[idx_base].append(i[0])
        for i in self.dataset_dst_ind.imgs:
            idx_base = self.base_mapping[self.idx_to_class_dst_ind[i[1]]]
            self.idx_imgs_dst_ind[idx_base].append(i[0])
    def __getitem__(self,index):

        dist = random.randint(1, 100)
        if(dist<6):
            opcion=1
        elif(dist<51):
            opcion = 2
        elif(dist<56):
            opcion = 3
        else:
            opcion = 4
        opcion = random.randint(1, 4)
        #opcion = random.choice([2, 4])
        label = torch.tensor([opcion])
        #0 = s, 1 = t
        op1 = 0
        op2 = 0
        same = 0
        if(opcion == 1):
            #sx, sx
            idx = random.choice(self.classes)
            idx1 = idx
            idx2 = idx
            img0_path = random.choice(self.idx_imgs_src[idx])
            img1_path = random.choice(self.idx_imgs_src[idx])
        if(opcion == 2):
            #tx, sx
            idx = random.choice(self.classes_ind)
            idx1 = idx
            idx2 = idx
            img0_path = random.choice(self.idx_imgs_src_ind[idx])
            img1_path = random.choice(self.idx_imgs_dst_ind[idx])
            op1 = 1
        if(opcion == 3):
            #sx, sy
            idx1 = random.choice(self.classes)
            idx2 = idx1
            while(idx2 == idx1):
                idx2 = random.choice(self.classes)
            img0_path = random.choice(self.idx_imgs_src[idx1])
            img1_path = random.choice(self.idx_imgs_src[idx2])
            same = 1
        if(opcion == 4):
            #sx, ty
            idx2 = random.choice(self.classes_ind)
            idx1 = idx2
            while(idx1 == idx2):
                idx1 = random.choice(self.classes_ind)
            img0_path = random.choice(self.idx_imgs_src_ind[idx1])
            img1_path = random.choice(self.idx_imgs_dst_ind[idx2])
            op2 = 1
            same = 1

        img0 = Image.open(img0_path)
        img1 = Image.open(img1_path)
        img0 = img0.convert("RGB")
        img1 = img1.convert("RGB")

        #idx1 = torch.tensor([idx1])
        #idx2 = torch.tensor([idx2])
        if(op1 == 0):
            img0 = self.transform_src(img0)
        if(op1 == 1):
            img0 = self.transform_dst(img0)
        if(op2 == 0):
            img1 = self.transform_src(img1)
        if(op2 == 1):
            img1 = self.transform_dst(img1)

        #label/=2
        label -= 1
        return img0, img1, idx1, idx2, label, op1, op2, same

    def __len__(self):
        return 2*min(len(self.dataset_src), len(self.dataset_dst))
        #return 4*min(len(self.dataset_src), len(self.dataset_dst))
        #return 120




def sample_src(src_path, transforms = None):
    dataset = datasets.ImageFolder(src_path, transforms)
    n=len(dataset)

    X=torch.Tensor(n,3,32,32)
    Y=torch.LongTensor(n)

    inds=torch.randperm(n)
    for i,index in enumerate(inds):
        x,y=dataset[index]
        X[i]=x
        Y[i]=y
    return X,Y


def sample_tgt(dst_path, transforms = None, n=1):
    dataset = datasets.ImageFolder(dst_path, transforms)
    X,Y=[],[]
    classes=10*[n]
    i=0
    while True:
        if len(X)==n*10:
            break
        x,y=dataset[i]
        if classes[y]>0:
            X.append(x)
            Y.append(y)
            classes[y]-=1
        i+=1

    assert (len(X)==n*10)
    return torch.stack(X,dim=0),torch.from_numpy(np.array(Y))

def create_groups(X_s,Y_s,X_t,Y_t,seed=1):
    #change seed so every time wo get group data will different in source domain,but in target domain, data not change
    torch.manual_seed(1 + seed)
    torch.cuda.manual_seed(1 + seed)


    n=X_t.shape[0] #10*shot


    #shuffle order
    classes = torch.unique(Y_t)
    classes=classes[torch.randperm(len(classes))]


    class_num=classes.shape[0]
    shot=n//class_num



    def s_idxs(c):
        idx=torch.nonzero(Y_s.eq(int(c)))

        return idx[torch.randperm(len(idx))][:shot*2].squeeze()
    def t_idxs(c):
        return torch.nonzero(Y_t.eq(int(c)))[:shot].squeeze()

    source_idxs = list(map(s_idxs, classes))
    target_idxs = list(map(t_idxs, classes))


    source_matrix=torch.stack(source_idxs)

    target_matrix=torch.stack(target_idxs)


    G1, G2, G3, G4 = [], [] , [] , []
    Y1, Y2 , Y3 , Y4 = [], [] ,[] ,[]


    for i in range(10):
        for j in range(shot):
            G1.append((X_s[source_matrix[i][j*2]],X_s[source_matrix[i][j*2+1]]))
            Y1.append((Y_s[source_matrix[i][j*2]],Y_s[source_matrix[i][j*2+1]]))
            G2.append((X_s[source_matrix[i][j]],X_t[target_matrix[i][j]]))
            Y2.append((Y_s[source_matrix[i][j]],Y_t[target_matrix[i][j]]))
            G3.append((X_s[source_matrix[i%10][j]],X_s[source_matrix[(i+1)%10][j]]))
            Y3.append((Y_s[source_matrix[i % 10][j]], Y_s[source_matrix[(i + 1) % 10][j]]))
            G4.append((X_s[source_matrix[i%10][j]],X_t[target_matrix[(i+1)%10][j]]))
            Y4.append((Y_s[source_matrix[i % 10][j]], Y_t[target_matrix[(i + 1) % 10][j]]))



    groups=[G1,G2,G3,G4]
    groups_y=[Y1,Y2,Y3,Y4]

    #make sure we sampled enough samples
    for g in groups:
        assert(len(g)==n)
    return groups,groups_y

def sample_groups(X_s,Y_s,X_t,Y_t,seed=1):
    print("Sampling groups")
    return create_groups(X_s,Y_s,X_t,Y_t,seed=seed)
