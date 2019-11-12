import torch
from torch import optim
import torch.nn as nn
import torchvision as tv
from torchvision import transforms

from models import fcn, unet, pspnet, dfn
from datasets.voc import to_rgb
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import numpy as np 

import time
from datetime import timedelta
import visdom

try:
    import nsml
    from nsml import Visdom
    USE_NSML = True
    print('NSML imported')
except ImportError:
    print('Cannot Import NSML. Use local GPU')
    USE_NSML = False

cudnn.benchmark = True # For fast speed

def imshow(self, img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
class Trainer:
    def __init__(self, train_data_loader, val_data_loader, config):     #gli passiamo i due dataset e la stringa di comandi
        self.cfg = config
        self.train_data_loader = train_data_loader          #associa il dataset train
        self.val_data_loader = val_data_loader              #associa il dataset val
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.build_model()      #crea il modello 

    def imshow(self, img):
        print("entrato")
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)    #ritorna il valore, 0 se minore di 0 e 1 se maggiore di 1. lo " NORMALIZZA " ( penso)

    def reset_grad(self):
        self.optim.zero_grad()   #resetta il gradiente

    def save_model(self, n_iter):   #come stra can si usa? che senno' mi tocca far training tutte le volte 
        checkpoint = {
            'n_iters' : n_iter + 1,
            'm_state_dict' : self.model.state_dict(),
            'optim' : self.optim.state_dict()
            }
        torch.save(checkpoint)

    def pixel_acc(self):
        """ Calculate accuracy of pixel predictions """
        pass

    def mean_IU(self):
        """ Calculate mean Intersection over Union """
        pass

    def build_model(self):                  
        if self.cfg.model == 'unet':        #noi usiamo U-NET, quindi bene  
            self.model = unet.UNet(num_classes=21, in_dim=3, conv_dim=64)
        elif self.cfg.model == 'fcn8':
            self.model = fcn.FCN8(num_classes=21)
        elif self.cfg.model == 'pspnet_avg':
            self.model = pspnet.PSPNet(num_classes=21, pool_type='avg')
        elif self.cfg.model == 'pspnet_max':
            self.model = pspnet.PSPNet(num_classes=21, pool_type='max')
        elif self.cfg.model == 'dfnet':
            self.model = dfn.SmoothNet(num_classes=21,
                                       h_image_size=self.cfg.h_image_size,      #settiamo la dimensione H
                                       w_image_size=self.cfg.w_image_size)      #settiamo la dimensione W
        self.optim = optim.Adam(self.model.parameters(),                        #usiamo adam per ottimizzazione stocastica come OPTIM, passangogli i parametri
                                lr=self.cfg.lr,                                 #settiamo il learning rate
                                betas=[self.cfg.beta1, self.cfg.beta2])     #le due Beta, cioe' la probabilita' di accettare l'ipotesi quando e' falsa  (coefficients used for computing running averages of gradient and its square )
        # Poly learning rate policy
        lr_lambda = lambda n_iter: (1 - n_iter/self.cfg.n_iters)**self.cfg.lr_exp       #ATTENZIONE: learning rate LAMBDA penso
        self.scheduler = LambdaLR(self.optim, lr_lambda=lr_lambda)
        self.c_loss = nn.CrossEntropyLoss().to(self.device)         #crossEntropy
        self.softmax = nn.Softmax(dim=1).to(self.device) # channel-wise softmax             #facciamo il softmax, cioe' prendiamo tutte le probabilita' e facciamo in modo che la loro somma sia 1

        self.n_gpu = torch.cuda.device_count()          #ritorna il numero di GPU a disposizione
        if self.n_gpu > 1:                                      
            print('Use data parallel model(# gpu: {})'.format(self.n_gpu))
            self.model = nn.DataParallel(self.model)        #implementa il parallelismo, se disponibile
        self.model = self.model.to(self.device)         

        if USE_NSML:        #non lo uso 
            self.viz = Visdom(visdom=visdom)

    def train_val(self):


        

        # Compute epoch's step size
        iters_per_epoch = len(self.train_data_loader.dataset) // self.cfg.train_batch_size      #divisione
        if len(self.train_data_loader.dataset) % self.cfg.train_batch_size != 0:
            iters_per_epoch += 1
        epoch = 1
        iters_per_epoch=5

        print('batch size {}\t'
                      'dataset size : [{}]\t'
                      'epoch : [{}]\t'
                      'datasetTime // batch : {}\t'.format(
                          self.cfg.train_batch_size,iters_per_epoch,len(self.train_data_loader.dataset), 
                      len(self.train_data_loader.dataset) // self.cfg.train_batch_size
                      ))

        torch.cuda.synchronize()  # parallel mode
        self.model.train()

        train_start_time = time.time()      #ciapiamo il tempo
        data_iter = iter(self.train_data_loader)  #iterabile dal dataset
        for n_iter in range(self.cfg.n_iters):      #da 0 al numero detto di n_iters
            self.scheduler.step()
            try:
                input, target = next(data_iter)     #passa il prossimo elemento dall'iteratore, input=immagine e target=mask
            except:
                data_iter = iter(self.train_data_loader)
                input, target = next(data_iter)

            input_var = input.clone().to(self.device)       #copiamo l'immagine nel device, che sarebbe la GPU
            target_var = target.to(self.device)             #prendiamo la maschera e la mettiamo indevice
            output = self.model(input_var)          #ora l'output diventa l'immagine
            #print( output.view(output.size(0), output.size(1), -1))#
            #print(target_var.view(target_var.size(0), -1))#
            loss = self.c_loss(output, target_var)      #loss sarebbe la Cross Entropy loss tra l'immagine e la mask

            self.reset_grad()           #resetta il gradiente
            loss.backward()         #va indietro del loss
            self.optim.step()   #update i parametri dopo l'ottimizzazione
            print('Done')#

            #output_label = torch.argmax(_output, dim=1)#

            if (n_iter + 1) % self.cfg.log_step == 0:
                seconds = time.time() - train_start_time        #secondi sono uguali al tempo trascorso meno quello di training, cioe' quanto tempo ci ha messo a fare il training
                elapsed = str(timedelta(seconds=seconds))
                print('Iteration : [{iter}/{iters}]\t'
                      'Time : {time}\t'
                      'Loss : {loss:.4f}\t'.format(
                      iter=n_iter+1, iters=self.cfg.n_iters,
                      time=elapsed, loss=loss.item()))          #controlla questo loss.item
                

            if (n_iter + 1) % iters_per_epoch == 0:     
                self.validate(epoch)                #se abbiamo finito un'epoca, validiamola e cambiamo epoca
                epoch += 1              #incrementiamo


    def validate(self, epoch):      

        self.model.eval()                                                #controlla EVAL
        val_start_time = time.time()                                     #prendiamo il tempo
        data_iter = iter(self.val_data_loader)                           #iteriamo su data_loader
        max_iter = len(self.val_data_loader)                            #prendiamo la dimensione
        for n_iter in range(max_iter):
            self.scheduler.step()
            try:
                input, target = next(data_iter)     #passa il prossimo elemento dall'iteratore, input=immagine e target=mask
            except:
                data_iter = iter(self.train_data_loader)
                input, target = next(data_iter)
            #n_iter =  0
            #input, target = next(data_iter)                         #andiamo avanti nell'iteratore

            input_var = input.clone().to(self.device)               #copiamo in device (GPU) input_var
            target_var = target.to(self.device)                 #copiamo la maschera in device

            output = self.model(input_var)                      #l'output diventa l'immagine (PENSO)
            _output = output.clone()                   
            output = output.view(output.size(0), output.size(1), -1)
            target_var = target_var.view(target_var.size(0), -1)
            loss = self.c_loss(output, target_var)                  #come prima

            output_label = torch.argmax(_output, dim=1)     #aggiunto
            #imshow(tv.utils.make_grid(input))      #aggiunto
            if (n_iter + 1) % self.cfg.log_step == 0:               #controlla questo LOG_STEP
                seconds = time.time() - val_start_time                  #tempo trascorso per validare
                elapsed = str(timedelta(seconds=seconds))
                print('### Validation\t'
                      'Iteration : [{iter}/{iters}]\t'
                    'Time : {time:}\t'
                    'Loss : {loss:.4f}\t'.format(
                    iter=n_iter+1, iters=max_iter,
                    time=elapsed, loss=loss.item()))

        if USE_NSML:                            #non lo usiamo
            ori_pic = self.denorm(input_var[0:4])
            self.viz.images(ori_pic, opts=dict(title='Original_' + str(epoch)))
            gt_mask = to_rgb(target_var[0:4])
            self.viz.images(gt_mask, opts=dict(title='GT_mask_' + str(epoch)))
            model_mask = to_rgb(output_label[0:4].cpu())
            self.viz.images(model_mask, opts=dict(title='Model_mask_' + str(epoch)))
