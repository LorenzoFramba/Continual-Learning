import numpy as np
import torch.nn as nn
import torch




def sigmLoss(output, target):
    loss = nn.BCELoss(reduce=False)
    loss1 = nn.CrossEntropyLoss()

    l1 = loss(output, target)
    l2 = loss1(output, target)
    print(l1,l2)
    