import argparse
import os

from torchvision import transforms
from torch.utils.data import DataLoader

from datasets.voc import VOC
from trainer import Trainer


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_loader(config):
    if config.dataset == 'voc':
        #root = os.path.join(config.path, 'train')
        transform = transforms.Compose([                        #unisce varie trasformazioni assieme
                transforms.Pad(10),                     #crea un paddig
                transforms.CenterCrop((config.h_image_size, config.w_image_size)),      #fa crop al centro, ma di quanto??
                transforms.ToTensor(),  #trasforma l'immagine in tensor ( con C x H x W, cioe Channels, Height and Width)
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  #normalizza il tensor nella media e deviazione standard
        ])

        train_data_set = VOC(root=config.path,                  #prendiamo il nostro dataset VOC e lo impostiamo come TRAIN
                             image_size=(config.h_image_size, config.w_image_size),  #h_image_size e w_image_size  sono 256 come argomento
                             dataset_type='train',
                             transform=transform)
        train_data_loader = DataLoader(train_data_set,                  #crea un dataset con un batch size
                                       batch_size=config.train_batch_size,  # 16 come argomento
                                       shuffle=True,
                                       drop_last=True,
                                       num_workers=config.num_workers, pin_memory=True)            #true vuol dire che per ogni epoca, il dataset viene mescolato

        val_data_set = VOC(root=config.path,            #prendiamo il nostro dataset VOC e lo impostiamo come TRAIN
                           image_size=(config.h_image_size, config.w_image_size),#h_image_size e w_image_size  sono 256 come argomento
                           dataset_type='val',
                           transform=transform)
        val_data_loader = DataLoader(train_data_set,  #crea un dataset con un batch size
                                     batch_size=config.val_batch_size,  # 16 come argomento
                                     shuffle=False,
                                     num_workers=config.num_workers, pin_memory=True) # For make samples out of various models, shuffle=False
    elif config.dataset == 'gta':
        # TODO:
        pass

    return train_data_loader, val_data_loader #ritorna i due vettori


def main(config):               #il config sarebbe il parser con tanti argomenti dei comandi
    import sys
    print(sys.version)
    make_dir(config.model_save_path)    #crea cartella del modello
    make_dir(config.sample_save_path) #crea cartella del sample

    if config.mode == 'train':
        train_data_loader, val_data_loader = get_loader(config) #associa ai due dataset i valori. presi dal config
        trainer = Trainer(train_data_loader=train_data_loader,  #fa partire il training, passando i due dataset
                         val_data_loader=val_data_loader,
                         config=config)
        trainer.train_val()                     #ora che la classe e' stata istanziata, fa partire il training


#qua andiamo solo a dichiarare i vari argomenti del config
if __name__ == '__main__':
    parser = argparse.ArgumentParser()  #libreria di linea di comando da string ad oggetti di python

                                #add_argument semplicemente popola il parser
    parser.add_argument('--mode', type=str, default='train', choices=['train'])
    parser.add_argument('--model', type=str, default='fcn8', choices=['fcn8', 'unet', 'pspnet_avg',
                                                                      'pspnet_max', 'dfnet'])
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

    # MISC


        #associamo a config il parser appena creato
    config = parser.parse_args()
    print(config)
    main(config)
