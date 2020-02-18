import torch
import numpy as np
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


# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)


palette = np.array([(0, 0, 0),
           (128, 0, 0),
           (0, 128, 0),
           (128, 128, 0),
           (0, 0, 128),
           (128, 0, 128),
           (0, 128, 128),
           (128, 128, 128),
           (64, 0, 0),
           (192, 0, 0),
           (64, 128, 0),
           (192, 128, 0),
           (64, 0, 128),
           (192, 0, 128),
           (64, 128, 128),
           (192, 128, 128),
           (0, 64, 0),
           (128, 64, 0),
           (0, 192, 0),
           (128, 192, 0),
           (0, 64, 128),
           (224, 224, 192)],
                     dtype=np.uint8)

def labelcolormap(N):
    if N == 22: # voc
        cmap = palette
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap


class Colorize(object):
    def __init__(self, n=22):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image