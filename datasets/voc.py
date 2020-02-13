import os
import sys
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms

#mezzi:  1 2 4 6 7 14 19
#animali: 3 8 10 12 13 15 17
#casa:  5 9 11 16 18 20
# palette = {
#0     'background':[0, 0, 0],           
#     'aeroplane':[128, 0, 0],      
#     'bicycle':[0, 128, 0],
#     'bird':[128, 128, 0],
#     'boat':[0, 0, 128],
#5     'bottle':[128, 0, 128],
#     'bus':[0, 128, 128],
#     'car':[128, 128, 128],
#8     'cat':[64, 0, 0],
#     'chair':[192, 0, 0],   *
#10     'cow':[64, 128, 0],
#     'diningtable':[192, 128, 0],
#     'dog':[64, 0, 128],
#     'horse':[192, 0, 128],
#     'motorbike':[64, 128, 128],
#15     'person':[192, 128, 128],
#     'pottedplant':[0, 64, 0], *
#     'sheep':[128, 64, 0],
#     'sofa':[0, 192, 0],   *
#     'train':[128, 192, 0],
#20     'tvmonitor':[0, 64, 128],
#     'void':[224, 224, 192]}

palette = [[0, 0, 0],
           [128, 0, 0],
           [0, 128, 0],
           [128, 128, 0],
           [0, 0, 128],
           [128, 0, 128],
           [0, 128, 128],
           [128, 128, 128],
           [64, 0, 0],
           [192, 0, 0],
           [64, 128, 0],
           [192, 128, 0],
           [64, 0, 128],
           [192, 0, 128],
           [64, 128, 128],
           [192, 128, 128],
           [0, 64, 0],
           [128, 64, 0],
           [0, 192, 0],
           [128, 192, 0],
           [0, 64, 128],
           [224, 224, 192]]

def to_mask(x):
    """
    input : (None, 3, H, W) - RGB value
    output : (None, H, W) - Label value
    """
    np_x = np.array(x)
    H, W, C = np_x.shape
    flatten_np_x = np_x.reshape(-1, C)
    empty = np.zeros_like(flatten_np_x)[:, 0]
    for i, it in enumerate(flatten_np_x):
        # Remove void part as background
        if list(it) == [224, 224, 192]:
            empty[i] = 0
        else:
            empty[i] = palette.index(list(it))

    mask = empty.reshape(H, W, 1).transpose(2, 0, 1)
    return torch.from_numpy(mask).squeeze().long()

def to_rgb(xs):
    """
    input : (None, H, W) - Label value
    output : (None, 3, H, W) - RGB value
    """
    rgbs = np.zeros((xs.size(0), xs.size(1), xs.size(2), 3))  #crea una matrice di zeri, con 4 assi
    for i, x in enumerate(xs):
        np_x = np.array(x)
        H, W = np_x.shape
        flatten_np_x = np_x.reshape(-1)
        expand_np_x = np_x.reshape(-1, 1).repeat(3, axis=-1) # (H, W, 3)
        for j in range(22):
            expand_np_x[np.where(flatten_np_x == j)] = palette[j]
        rgbs[i] = expand_np_x.reshape(H, W, 3)
    rgbs = torch.from_numpy(rgbs.transpose(0, 3, 1, 2))
    return rgbs

def make_path(root):                #prende le imagini e le maschere e le unisce in un'unico item.
    train_items = []            #vettore per train
    val_items = []              #vettore per validation

    img_path = os.path.join(root, 'VOC2012', 'JPEGImages')
    mask_path = os.path.join(root, 'VOC2012', 'SegmentationClass')
    train_data_list = [l.strip('\n') for l in open(os.path.join(root, 'VOC2012',
                'ImageSets', 'Segmentation', 'train.txt')).readlines()]
    val_data_list   = [l.strip('\n') for l in open(os.path.join(root, 'VOC2012',
                'ImageSets', 'Segmentation', 'train.txt')).readlines()]  #'train.txt'  -> val.txt ?

    for it in train_data_list:
        item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
        #print("item",item)
        train_items.append(item)          #aggiunge elemento in train

    for it in val_data_list:
        item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
        val_items.append(item)          #aggiunge elemento in validating


    return train_items, val_items       #ritorna i due vettori



class VOC(data.Dataset):
    def __init__(self, root, image_size, dataset_type, transform=None, target_transform=to_mask):
        """
        root - parent of data file
        dataset_type - ['train', 'val']
        """
        assert dataset_type in ['train', 'val'], 'dataset_type should be in train/val'
        self.train_items, self.val_items = make_path(root)
        self.h_image_size, self.w_image_size = image_size[0], image_size[1]
        self.dataset_type = dataset_type
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        if self.dataset_type == 'train':            #se l'elemento del vettore e' train
            name = self.train_items[i]              #name e' l'elemento corrispondente all'index
        elif self.dataset_type == 'val':
            name = self.val_items[i]

        image = Image.open(name[0]).convert('RGB')  #image
        mask = Image.open(name[1]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = transforms.Pad(10)(mask)  #aggiunge un padding di 10 pixels
            mask = transforms.CenterCrop((self.h_image_size, self.w_image_size))(mask)  #aggiunge un crop
            mask = self.target_transform(mask)

        return image, mask

    def __len__(self):
        if self.dataset_type == 'train':
            return len(self.train_items)
        elif self.dataset_type == 'val':
            return len(self.val_items)

if __name__ == "__main__":
    print(sys.path[0])
    transform = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])
    data_set = VOC(root="./datasets", image_size=128, dataset_type='train', transform=transform, target_transform=to_mask)
    for data in data_set:
        print(np.array(data[0]).shape, np.array(data[1]).shape)
    # np.set_printoptions(threshold=np.nan)
    print(data_set[0][1].type())
    # print(data_set[0][1])
