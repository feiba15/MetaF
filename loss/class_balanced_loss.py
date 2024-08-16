
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class CE_weight(nn.Module):
    '''
    Balanced-Weight Cross-Entropy loss function
    '''

    def __init__(self, cls_num_list, E1 = 200, E2 = 50, E = 100):
        super(CE_weight, self).__init__()
        self.cls_num_list = cls_num_list
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)

        # weight of each class for imbalance dataset
        weight = torch.cuda.FloatTensor(1.0 / cls_num_list)
        self.weight = (weight / weight.sum()) * len(cls_num_list)

        #hyper-parameters of stages
        self.E1 = int(E1)
        self.E2 = int(E2)
        self.E = int(E)


    def forward(self, x, target, e, f1_score = [1,1,1,1]):
        '''
        :param x: input
        :param target: label
        :param e: current epoch
        :param f1_score: f1 score on validation set
        :return: loss
        '''
        if e <= self.E1:
            return F.cross_entropy(x, target)

        # if e > self.E1:
        #     now_power = (e-self.E1) / (self.E2-self.E1)
        #     per_cls_weights = [torch.pow(num, now_power) for num in self.weight]
        #     per_cls_weights = torch.cuda.FloatTensor(per_cls_weights)
        #     return F.cross_entropy(x,target, weight=per_cls_weights)

        if e > self.E1 and e <= self.E2:
            now_power = (e-self.E1) / (self.E2-self.E1)
            per_cls_weights = [torch.pow(num, now_power) for num in self.weight]
            per_cls_weights = torch.cuda.FloatTensor(per_cls_weights)
            return F.cross_entropy(x,target, weight=per_cls_weights)

        else:
            f1_score = torch.cuda.FloatTensor(f1_score)
            weight = torch.cuda.FloatTensor(1.0 / f1_score)
            self.weight = (weight / weight.sum()) * len(self.cls_num_list)
            now_power = (e - self.E2) / (self.E - self.E2)
            per_cls_weights = [torch.pow(num, now_power) for num in self.weight]
            per_cls_weights = torch.cuda.FloatTensor(per_cls_weights)
            return F.cross_entropy(x, target, weight=per_cls_weights)

def focal_loss(labels, logits, alpha, gamma): 
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss



def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    return cb_loss

if __name__ == '__main__':
    no_of_classes = 5
    logits = torch.rand(10,no_of_classes).float()
    labels = torch.randint(0,no_of_classes, size = (10,))
    beta = 0.9999
    gamma = 2.0
    samples_per_cls = [2,3,1,2,2]
    loss_type = "focal"
    cb_loss = CB_loss(labels, logits, samples_per_cls, no_of_classes,loss_type, beta, gamma)
    print(cb_loss)



