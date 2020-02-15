import argparse
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from datasets.voc import VOC
from trainer import Trainer
import numpy as np 
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import csv
import torchvision as tv



def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

########### Config Loader ###########

def get_loader(config):
        transform = transforms.Compose([                                #unisce varie trasformazioni assieme
                transforms.Pad(10),                                     #crea un paddig
                transforms.CenterCrop((config.h_image_size, config.w_image_size)),      #fa crop al centro, ma di quanto??
                transforms.ToTensor(),                                  #trasforma l'immagine in tensor ( con C x H x W, cioe Channels, Height and Width)
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  #normalizza il tensor nella media e deviazione standard
        ])
        train_list="ciao"
        val_list="ciao"
        if config.step == 'default':   
                train_list = "train.txt"
                val_list = "val.txt"    
        if config.step == 'split_1':   
                train_list = "train_split_1.txt"
                val_list = "val_split_1.txt"
        if config.step == 'split_2': 
                train_list = "train_split_2.txt"
                val_list = "val_split_2.txt"
        if config.step == 'split_3': 
                train_list = "train_split_3.txt"
                val_list = "val_split_3.txt"
        train_data_set = VOC(root=config.path,                          #prendiamo il nostro dataset VOC e lo impostiamo come TRAIN
                                image_size=(config.h_image_size, config.w_image_size),  #h_image_size e w_image_size  sono 256 come argomento
                                dataset_type='train',
                                transform=transform, train_list=train_list ,val_list=val_list )


                
        train_data_loader_1 = DataLoader(train_data_set,                  #crea un dataset con un batch size
                                        batch_size=config.train_batch_size,  # 16 come argomento
                                        shuffle=False,    #METTILO true
                                        drop_last=True,
                                        num_workers=config.num_workers, pin_memory=True) 
        
        val_data_set = VOC(root=config.path,                            #prendiamo il nostro dataset VOC e lo impostiamo come TRAIN
                                image_size=(config.h_image_size, config.w_image_size),#h_image_size e w_image_size  sono 256 come argomento
                                dataset_type='val',
                                transform=transform, train_list=train_list, val_list=val_list )

        val_data_loader_1 = DataLoader(val_data_set,                    #crea un dataset con un batch size
                                batch_size=config.val_batch_size,  #16 come argomento
                                shuffle=False,
                                drop_last=True,
                                num_workers=config.num_workers, pin_memory=True) # For make samples out of various models, shuffle=False

        
        return train_data_loader_1, val_data_loader_1

def separa(train_data_loader, val_data_loader):
        train_mezzi_data = []
        train_casa_data = []
        train_animali_data = []
           
        train_custom_data = []
        train_con_cose_data = []


        val_mezzi_data = []
        val_casa_data = []
        val_animali_data = []
           
        val_custom_data = []
        val_con_cose_data = []


        root = config.path
            

        img_path = os.path.join(root, 'VOC2012', 'JPEGImages')
        mask_path = os.path.join(root, 'VOC2012', 'SegmentationClass')
        train_data_list = [l.strip('\n') for l in open(os.path.join(root, 'VOC2012',
                            'ImageSets', 'Segmentation', 'train.txt')).readlines()]
        val_data_list = [l.strip('\n') for l in
                               open(os.path.join(root, 'VOC2012',
                                                 'ImageSets', 'Segmentation',
                                                 'val.txt')).readlines()]

        a = 0
        for i, (image, mask) in enumerate(iter(train_data_loader)):                
                print("TRAIN numero", i)
                    #mezzi:  1 2 4 6 7 14 19
                    #animali: 3 8 10 12 13 15 17
                    #casa:  5 9 11 16 18 20
                for I in range(config.train_batch_size):

                        nomeFoto = train_data_list[a]
                        item = (os.path.join(img_path, nomeFoto + '.jpg'), os.path.join(mask_path, nomeFoto + '.png'))

                        a+=1
                        print("TRAIN ITERAZIONE: ", I, " su ",config.train_batch_size )
                        out = mask[I].numpy().flatten()   
                        lista = np.unique(out)

                        mezzi_   = [0,21, 1, 2, 4, 6, 7, 14, 19]
                        animali_ = [0,21, 3 ,8 ,10 ,12 ,13 ,15, 17]
                        casa_    = [0,21, 5 ,9 ,11, 16, 18, 20]
                        mezzi   =  all(elem in mezzi_  for elem in lista)
                        animali =  all(elem in animali_  for elem in lista)
                        casa    =  all(elem in casa_  for elem in lista)
                        
                        if(mezzi):
                            print("TRAIN  ERA UN MEZZO ")
                            train_mezzi_data.append(nomeFoto)
                            #tv.utils.save_image(image,os.path.join(config.sorted_save_path,"mezzi",f"input_{i}_{I}.jpg"),normalize=True, range=(-1,1))  
                        
                        elif(animali):
                            print("TRAIN  ERA UN ANIMALE ")
                            train_animali_data.append(nomeFoto)
                            #tv.utils.save_image(image,os.path.join(config.sorted_save_path,"animali",f"input_{i}_{I}.jpg"),normalize=True, range=(-1,1))  
                        elif(casa):
                            print("TRAIN  ERA IN CASA ")
                            train_casa_data.append(nomeFoto)
                            #tv.utils.save_image(image,os.path.join(config.sorted_save_path,"casa",f"input_{i}_{I}.jpg"),normalize=True, range=(-1,1))  
                        else:
                            print("TRAIN immagine",I," in batch ", i ," non appartiene a nessun gruppo")  
                            train_con_cose_data.append(lista)
                            train_custom_data.append(nomeFoto)
                            print(" ed ha ste classi ",train_con_cose_data)
                            train_con_cose_data.clear()
                            #tv.utils.save_image(image,os.path.join(config.sorted_save_path,"random",f"input_{i}_{I}.jpg"),normalize=True, range=(-1,1))  
                
        b = 0
        
        for i, (image, mask) in enumerate(iter(val_data_loader)):
                
                  print("VAL numero", i)
                  for I in range(config.val_batch_size):
                        if(b<1449):
                                nomeFoto = val_data_list[b]
                                print("numero b: ", b)
                                item = (os.path.join(img_path, nomeFoto + '.jpg'), os.path.join(mask_path, nomeFoto + '.png'))
                                
                                b+=1
                                print("VAL ITERAZIONE: ", I, " su ",config.val_batch_size )
                                out = mask[I].numpy().flatten()   
                                lista = np.unique(out)

                                mezzi_   = [0,21, 1, 2, 4, 6, 7, 14, 19]
                                animali_ = [0,21, 3 ,8 ,10 ,12 ,13 ,15, 17]
                                casa_    = [0,21, 5 ,9 ,11, 16, 18, 20]
                                mezzi   =  all(elem in mezzi_  for elem in lista)
                                animali =  all(elem in animali_  for elem in lista)
                                casa    =  all(elem in casa_  for elem in lista)
                                
                                if(mezzi):
                                        print(" VAL ERA UN MEZZO ")
                                        val_mezzi_data.append(nomeFoto)
                                        #tv.utils.save_image(image,os.path.join(config.sorted_save_path,"mezzi",f"input_{i}_{I}.jpg"),normalize=True, range=(-1,1))  
                                
                                elif(animali):
                                        print(" VAL ERA UN ANIMALE ")
                                        val_animali_data.append(nomeFoto)
                                        #tv.utils.save_image(image,os.path.join(config.sorted_save_path,"animali",f"input_{i}_{I}.jpg"),normalize=True, range=(-1,1))  
                                elif(casa):
                                        print(" VAL ERA IN CASA ")
                                        val_casa_data.append(nomeFoto)
                                        #tv.utils.save_image(image,os.path.join(config.sorted_save_path,"casa",f"input_{i}_{I}.jpg"),normalize=True, range=(-1,1))  
                                else:
                                        print("VAL immagine",I," in batch ", i ," non appartiene a nessun gruppo")  
                                        val_con_cose_data.append(lista)
                                        val_custom_data.append(nomeFoto)
                                        print(" ed ha ste classi ",val_con_cose_data)
                                        val_con_cose_data.clear()
                                        #   tv.utils.save_image(image,os.path.join(config.sorted_save_path,"random",f"input_{i}_{I}.jpg"),normalize=True, range=(-1,1))  

                                
                        
                        
        print("TRAIN MEZZI:" ,len(train_mezzi_data)) 
        print("TRAIN ANIMALI:", len(train_animali_data)) 
        print("TRAIN CASE:", len(train_casa_data)) 
        print("TRAIN A CASO:", len(train_custom_data)) 
        print("VAL MEZZI:" ,len(val_mezzi_data)) 
        print("VAL ANIMALI:", len(val_animali_data)) 
        print("VAL CASE:", len(val_casa_data)) 
        print("VAL A CASO:", len(val_custom_data)) 


        with open(os.path.join(root, 'VOC2012',
                            'ImageSets', 'Segmentation', 'train_split_1.txt'), 'w') as file_handler:

                file_handler.write("\n".join(str(item) for item in train_mezzi_data))
        with open(os.path.join(root, 'VOC2012',
                            'ImageSets', 'Segmentation', 'train_split_2.txt'), 'w') as file_handler:

                file_handler.write("\n".join(str(item) for item in train_animali_data))
        with open(os.path.join(root, 'VOC2012',
                            'ImageSets', 'Segmentation', 'train_split_3.txt'), 'w') as file_handler:

                file_handler.write("\n".join(str(item) for item in train_casa_data))
        with open(os.path.join(root, 'VOC2012',
                            'ImageSets', 'Segmentation', 'train_split_4.txt'), 'w') as file_handler:

                file_handler.write("\n".join(str(item) for item in train_custom_data))


        with open(os.path.join(root, 'VOC2012',
                            'ImageSets', 'Segmentation', 'val_split_1.txt'), 'w') as file_handler:

                file_handler.write("\n".join(str(item) for item in val_mezzi_data))
        with open(os.path.join(root, 'VOC2012',
                            'ImageSets', 'Segmentation', 'val_split_2.txt'), 'w') as file_handler:

                file_handler.write("\n".join(str(item) for item in val_animali_data))
        with open(os.path.join(root, 'VOC2012',
                            'ImageSets', 'Segmentation', 'val_split_3.txt'), 'w') as file_handler:

                file_handler.write("\n".join(str(item) for item in val_casa_data))
        with open(os.path.join(root, 'VOC2012',
                            'ImageSets', 'Segmentation', 'val_split_4.txt'), 'w') as file_handler:

                file_handler.write("\n".join(str(item) for item in val_custom_data))           


def main(config):                                                       #il config sarebbe il parser con tanti argomenti dei comandi
    import sys
    print(sys.version)
    make_dir(config.model_save_path+"/model_default")                                    #crea cartella del modello
    make_dir(config.model_save_path+"/models_split1") 
    make_dir(config.model_save_path+"/models_split2") 
    make_dir(config.model_save_path+"/models_split3") 

    make_dir(config.sorted_save_path)
    make_dir(config.sample_save_path+"/samples_default")                                   #crea cartella del sample
    make_dir(config.sample_save_path+"/samples_split1")
    make_dir(config.sample_save_path+"/samples_split2")
    make_dir(config.sample_save_path+"/samples_split3")
    for folder in ["inputs","ground_truth","generated"]:                #tra i vari folders delle foto
        make_dir(os.path.join(config.sample_save_path+"/samples_default", folder))         #crea le cartelle in questione
    for folder in ["inputs","ground_truth","generated"]:                #tra i vari folders delle foto
        make_dir(os.path.join(config.sample_save_path+"/samples_split1", folder)) 
    for folder in ["inputs","ground_truth","generated"]:                #tra i vari folders delle foto
        make_dir(os.path.join(config.sample_save_path+"/samples_split1", folder)) 
    for folder in ["inputs","ground_truth","generated"]:                #tra i vari folders delle foto
        make_dir(os.path.join(config.sample_save_path+"/samples_split1", folder))     
    for folder in ["mezzi","animali","casa","random"]:                #tra i vari folders delle foto
        make_dir(os.path.join(config.sorted_save_path, folder))
    if config.mode == 'train':
        if config.step == 'default':   
                config.train_list = "train.txt"
                config.val_list = "val.txt"    
        if config.step == 'split_1':   
                config.train_list = "train_split_1.txt"
                config.val_list = "val_split_1.txt"
        if config.step == 'split_2': 
                config.train_list = "train_split_2.txt"
                config.val_list = "val_split_2.txt"
        if config.step == 'split_3': 
                config.train_list = "train_split_3.txt"
                config.val_list = "val_split_3.txt"
        train_data_loader_1, val_data_loader_1= get_loader(config)         #associa ai due dataset i valori. presi dal config
        trainer_1 = Trainer(train_data_loader=train_data_loader_1,          #fa partire il training, passando i due dataset
                            val_data_loader=val_data_loader_1,
                            config=config)
        trainer_1.train_val()
    if config.mode == "split_dataset":
        config.train_list = "train.txt"
        config.val_list = "val.txt"
        train_data_loader_1, val_data_loader_1 = get_loader(
            config)  # associa ai due dataset i valori. presi dal config
        separa(train_data_loader_1, val_data_loader_1)                                            #ora che la classe e' stata istanziata, fa partire il training

########### Config Parameters ###########

if __name__ == '__main__':
    parser = argparse.ArgumentParser()                                  #libreria di linea di comando da string ad oggetti di python

                                                                        #add_argument semplicemente popola il parser
    parser.add_argument('--step', type=str, default='split_1', choices=['split_1', 'split_2','split_3','default'])
    parser.add_argument('--mode', type=str, default='train', choices=['train', "split_dataset"])
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'fcn8', 'pspnet_avg', 'pspnet_max', 'dfnet'])
    parser.add_argument('--dataset', type=str, default='voc', choices=['voc'])
   

    # Training setting
    parser.add_argument('--n_iters', type=int, default=10000)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)               #learning rate
    parser.add_argument('--lr_exp', type=float, default=0.9)
    parser.add_argument('--beta1', type=float, default=5e-1)            #the probability of of accepting the null hypothesis when it’s false.
    parser.add_argument('--beta2', type=float, default=0.99)            #the probability of of accepting the null hypothesis when it’s false.
    parser.add_argument('--h_image_size', type=int, default=512)
    parser.add_argument('--w_image_size', type=int, default=256)
    # Hyper parameters
    #TODO

    # Path
    parser.add_argument('--model_save_path', type=str, default='./model')
    parser.add_argument('--sample_save_path', type=str, default='./sample')
    parser.add_argument('--sorted_save_path', type=str, default='./sorting')
    parser.add_argument('--path', type=str, default='./dataset')

    # Logging setting
    parser.add_argument('--log_step', type=int, default=1)
    parser.add_argument('--val_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10, help='Saving epoch')
    parser.add_argument('--sample_save_step', type=int, default=10, help='Saving epoch')
    parser.add_argument('--continue_train', action='store_true',
                             help='continue training: load the latest model')
    parser.add_argument('--which_epoch', type=str, default='latest',
                             help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument("--num_workers", type=int, default=4, help="num of threads for multithreading")
    parser.add_argument("--train_list", type=str, default="train_split_1.txt")
    parser.add_argument("--val_list", type=str, default="val_split_1.txt")

    # MISC

    ########### parsing to config the created parser ###########
    config = parser.parse_args()
    print(config)
    main(config)
