import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.optim.lr_scheduler as lr_scheduler
from adv import VATLoss
from cross import CROSSLoss
from model import Network
import copy
from scipy import sparse
# Torchvison
class al:
    def __init__(self, K=50, model=None):

        self.K = K
        self.model=None
def al_selection(model,un_loader=None):
    x = None
    y_l = None
    for img, label, sex, age in un_loader:
        img = img.cuda()
        label = label.cuda()
        with torch.no_grad():
            _,_, feature_last = model(img)
            if y_l is not None:
                y_l = torch.cat((y_l, label),0)
                # print(y_l.shape)
            else:
                y_l = label

            if x is not None:
                x = torch.cat((x,feature_last),0)
                # print(x.shape)
            else:
                x=feature_last
    
    features = x.cpu()   #n*m
    vus_ac=vus(model,features)
    bus_ac=bus(model,features)
    return vus_ac, bus_ac #list(set(vus_ac).union(set(bus_ac)))

def bus(model, features): #5 15
    cross_loss = CROSSLoss()
    result = torch.tensor(cross_loss(model, features))
    _, querry_indices = torch.topk(result, int(30), largest=False)
    return querry_indices

def vus(model,features):  #210  630
         vat_loss = VATLoss()
         lds, lds_each = vat_loss(model, features)
         lds_each = lds_each.view(-1)
         _, querry_indices = torch.topk(lds_each, int(30))
         querry_indices = querry_indices.cpu()
         return querry_indices

