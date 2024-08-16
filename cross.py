import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Network
import numpy as np

class CROSSLoss(nn.Module):

    def __init__(self):
        """
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(CROSSLoss, self).__init__()

    def similarity(self, x):
        x_copy = x.clone()
        # x = x.unsqueeze(1)
        # x = x.repeat(1, x.size(0), 1)
        result = []
        # print(x.shape)
        # print(x_copy.shape)
        # exit(0)
        for index in range(x.size(0)):
            # output = F.cosine_similarity(x[index:index+1], x_copy)
        #     print(index)
            result.append(torch.sum(F.cosine_similarity(x[index:index+1], x_copy))
                          /x.size(0))
        result_tensor = torch.tensor(result)
        return result

    def forward(self, model, x):
        with torch.no_grad():
            for name, module in model.module.named_modules():
                if name == "model.classifier":
                    model = module.cpu()
                    break

        # pred = F.softmax(model(x), dim=1)
        pred = F.softmax(model(x), dim=1)
        pred_log = torch.log(pred)
        loss = torch.zeros(pred.size(0), pred.size(1))
        for dim0 in range(pred.size(0)):
            for dim1 in range(pred.size(1)):
                loss[dim0][dim1] = pred[dim0][dim1]*pred_log[dim0][dim1]
        loss = torch.sum(loss, dim=-1).detach().tolist()

        # # criterion = nn.CrossEntropyLoss()
        # # loss = np.array(criterion(pred, pred)).tolist()  #n*1
        weight = self.similarity(x)

        for index in range(len(loss)):
            loss[index] = loss[index] * weight[index]

        return loss



