import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda=True if torch.cuda.is_available() else False
device=torch.device('cuda:0') if use_cuda else torch.device('cpu')

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        #if(label == 0):
        #    loss_contrastive = torch.pow(euclidean_distance, 2)
        #else:
        label = label.to(device)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive
        
class SpecLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(SpecLoss, self).__init__()
        self.margin = margin
        self.loss_fn=torch.nn.CrossEntropyLoss()

    def forward(self, output, label, domains, analyze):
        mapa = domains==analyze
        output = output[mapa]
        label = label[mapa]
        if(len(label)==0):
            return 0
        val = self.loss_fn(output, label)
        return val
