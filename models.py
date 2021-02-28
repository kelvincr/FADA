import torch
import torch.nn as nn
from torch.nn import init
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import numpy as np


class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule,self).__init__()
        self.name = 'result/model.pth'
    def save(self, path = None):
        torch.save(self.state_dict(),self.name)
        return self.name
    def load(self,path = None):
        self.load_state_dict(torch.load(self.name))

class DCD(BasicModule):
    def __init__(self,h_features=64,input_features=128):
        super(DCD,self).__init__()
        self.name = 'result/datapropio_digits/dcd.pth'
        self.fc1=nn.Linear(input_features,h_features)
        self.fc2=nn.Linear(h_features,h_features)
        self.fc3=nn.Linear(h_features,4)

    def forward(self,inputs):
        out=F.relu(self.fc1(inputs))
        out=self.fc2(out)
        return F.softmax(self.fc3(out),dim=1)




class DCDPro(BasicModule):
    def __init__(self):
        super(DCDPro,self).__init__()
        self.name = 'result/dcdpro_r18.pth'
        self.fc00=nn.Linear(4096,  2048)  #para r50 e inception
        self.fc0=nn.Linear(2048,  1024)   #para r50 e inception
        self.fc1=nn.Linear(1024,512)
        self.fc2=nn.Linear(512,120)
        self.fc3=nn.Linear(120,10)
        self.fc4=nn.Linear(10,4)  #4

    def forward(self,x):
        out = x
        out=F.relu(self.fc00(out))
        out=F.relu(self.fc0(out))
        out=F.relu(self.fc1(out))
        out=F.relu(self.fc2(out))
        out=F.relu(self.fc3(out))
        out=F.softmax(self.fc4(out),dim=1)
        return out

class Classifier(BasicModule):
    def __init__(self,input_features=64):
        super(Classifier,self).__init__()
        self.name = 'result/datapropio_digits/classifier.pth'
        self.fc=nn.Linear(input_features,10)

    def forward(self,input):
        return F.softmax(self.fc(input),dim=1)

class EncoderPro(BasicModule):

    def __init__(self):
        super(EncoderPro, self).__init__()
        self.name = 'result/encoderpro.pth'
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ClassifierPro(BasicModule):

    def __init__(self):
        super(ClassifierPro, self).__init__()
        self.name = 'result/classifierpro_r18.pth'
        self.fc0=nn.Linear(2048, 997) #para r50 e inception
        #self.fc1=nn.Linear(1024,512)  #para r50 e inception
        #self.fc2=nn.Linear(512,120)
        #self.fc3=nn.Linear(120,10)

    def forward(self, x):
        out = x
        out=self.fc0(out)
        #out=F.relu(self.fc1(out))
        #out=F.relu(self.fc2(out))
        #out=F.softmax(self.fc3(out),dim=1)
        return out

class Encoder(BasicModule):
    def __init__(self):
        super(Encoder,self).__init__()
        self.name = 'result/encoder.pth'
        self.conv1=nn.Conv2d(3,6,5)
        #self.conv1=nn.Conv2d(3,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc0=nn.Linear(400,256)
        self.fc1=nn.Linear(256,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,64)

    def forward(self,input):
        out=F.relu(self.conv1(input))
        out=F.max_pool2d(out,2)
        out=F.relu(self.conv2(out))
        out=F.max_pool2d(out,2)
        #print(out.size())
        out=out.view(out.size(0),-1)
        out=F.relu(self.fc0(out))
        out=F.relu(self.fc1(out))
        out=F.relu(self.fc2(out))
        out=self.fc3(out)


        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.dfc3 = nn.Linear(512, 4096)
        self.bn3 = nn.BatchNorm1d(4096)
        self.dfc2 = nn.Linear(4096, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.dfc1 = nn.Linear(4096,256 * 6 * 6)
        self.bn1 = nn.BatchNorm1d(256*6*6)
        self.upsample1=nn.Upsample(scale_factor=2)
        self.dconv5 = nn.ConvTranspose2d(256, 256, 3, padding = 0)
        self.dconv4 = nn.ConvTranspose2d(256, 384, 3, padding = 1)
        self.dconv3 = nn.ConvTranspose2d(384, 192, 3, padding = 1)
        self.dconv2 = nn.ConvTranspose2d(192, 64, 5, padding = 2)
        self.dconv1 = nn.ConvTranspose2d(64, 3, 12, stride = 4, padding = 4)
    def forward(self,x):#,i1,i2,i3):
        x = self.dfc3(x)
        bs = x.size(0)
		#x = F.relu(x)
        x = F.relu(self.bn3(x))

        x = self.dfc2(x)
        x = F.relu(self.bn2(x))
        #x = F.relu(x)
        x = self.dfc1(x)
        x = F.relu(self.bn1(x))
        #x = F.relu(x)
        #print(x.size())
        x = x.view(bs,256,6,6)
        #print (x.size())
        x=self.upsample1(x)
        #print x.size()
        x = self.dconv5(x)
        #print x.size()
        x = F.relu(x)
        #print x.size()
        x = F.relu(self.dconv4(x))
        #print x.size()
        x = F.relu(self.dconv3(x))
        #print x.size()
        x=self.upsample1(x)
        #print x.size()
        x = self.dconv2(x)
        #print x.size()
        x = F.relu(x)
        x=self.upsample1(x)
        #print x.size()
        x = self.dconv1(x)
        #print x.size()
        x = torch.sigmoid(x)
        #print x
        return x

class SimpleNet(BasicModule):
    def __init__(self,input_features=64, output_features = 10):
        super(SimpleNet,self).__init__()
        self.fc=nn.Linear(input_features,output_features)

    def forward(self,input):
        return F.softmax(self.fc(input),dim=1)
class TaxonNet(BasicModule):

    def __init__(self, outputs):
        super(TaxonNet, self).__init__()
        self.name = 'result/classifierpro_r18.pth'
        self.fc0=nn.Linear(2048, outputs) #para r50 e inception
        #self.fc1=nn.Linear(1024,512)  #para r50 e inception
        #self.fc2=nn.Linear(512,120)
        #self.fc3=nn.Linear(120,10)

    def forward(self, x):
        out = x
        out=self.fc0(out)
        return out
