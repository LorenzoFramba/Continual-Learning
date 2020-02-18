import numpy as np
import torch
from sklearn.metrics import confusion_matrix as cm


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum().item()
    valid_sum = valid.sum().item()
    assert valid_sum != 0
    acc = float(acc_sum) / (valid_sum)
    return acc, valid_sum


SMOOTH = 1e-6


def intersectionAndUnion_torch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum(
        (1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum(
        (1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (
                union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0,
                              10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()              
    imLab = np.asarray(imLab).copy()
    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)
    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram( intersection, bins=numClass, range=(1, numClass))
    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection
    return (area_intersection, area_union)


def class_accuracy(preds, label, confusion_matrix, labels = range(0,22)):
    res = cm(label.view(-1),preds.view(-1), labels)
    confusion_matrix += torch.from_numpy(res).float()
    return confusion_matrix