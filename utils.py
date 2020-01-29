import torch
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.confusion_matrix = None
    def initialize(self, val, weight, classes = 22):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True
        self.confusion_matrix = torch.zeros(classes, classes)
    def update(self, val, weight=1, classes = 22):
        if not self.initialized:
            self.initialize(val, weight, classes)
        else:
            self.add(val, weight)
    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count
    def value(self):
        return self.val
    def average(self):
        return self.avg
    def get_confusion_matrix(self):
        return self.confusion_matrix
    def update_confusion_matrix(self, confusion_matrix):
        self.confusion_matrix = confusion_matrix