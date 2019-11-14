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
import os, sys
import time
from datetime import timedelta

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
        self.start_epoch = 0
        self.build_model()      #crea il modello

    def imshow(self, img):
        print("entrato")
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def denorm(self, x):
        """
        Pier
        :param x: a normalized tensor in range [-1,1]
        :return: a "denormalized" tensor in range [0,1]
        """
        out = (x + 1) / 2
        return out.clamp_(0, 1)    #ritorna il valore, 0 se minore di 0 e 1 se maggiore di 1. lo " NORMALIZZA " ( penso)

    def reset_grad(self):
        self.optim.zero_grad()   #resetta il gradiente

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids,
                     epoch, optimizer, scheduler):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.cfg.model_save_path, save_filename)
        print(save_path)
        state = {
            "epoch": epoch + 1,
            "model_state": network.cpu().state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict()
        }
        torch.save(state, save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda()

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label,
                     epoch, optimizer, scheduler, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_dir = self.cfg.model_save_path
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if network_label == 'G':
                raise ('Generator must exist!')
        else:
            # network.load_state_dict(torch.load(save_path))
            try:
                checkpoint = torch.load(save_path)
                network.load_state_dict(checkpoint["model_state"])
                epoch = checkpoint["epoch"]
                optimizer.load_state(checkpoint["optimizer_state"])
                scheduler.load_state(checkpoint["scheduler_state"])
                print("Load model Done!")
            except:
                print("Error during the load of the model")

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
        self.c_loss = nn.CrossEntropyLoss().to(self.device)         #crossEntropy ! muove il modello nella GPU
        self.softmax = nn.Softmax(dim=1).to(self.device) # channel-wise softmax             #facciamo il softmax, cioe' prendiamo tutte le probabilita' e facciamo in modo che la loro somma sia 1

        self.n_gpu = torch.cuda.device_count()          #ritorna il numero di GPU a disposizione
        if self.cfg.continue_train:
            self.load_network(self.model, "UNET_VOC", self.cfg.which_epoch,
                              self.start_epoch, self.optim, self.scheduler)
        if self.n_gpu > 1:
            print('Use data parallel model(# gpu: {})'.format(self.n_gpu))
            self.model = nn.DataParallel(self.model)        #implementa il parallelismo, se disponibile
        self.model = self.model.to(self.device)
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True



    def train_val(self):
        best_loss = 100000000.0
        since = time.time()
        iters_per_epoch = len(self.train_data_loader.dataset) // self.cfg.train_batch_size      #divisione tenendo interi
        epoch = self.start_epoch

        print(f"batch size {self.cfg.train_batch_size} dataset size : [{len(self.train_data_loader.dataset)}]"
              f" epoch : [{self.cfg.n_iters}]"
              f" iterations per epoch: {iters_per_epoch}")
        data_iter = iter(self.train_data_loader)

        for epoch in range(self.cfg.n_iters):    #numero di barchs
            print('Epoch {}/{}'.format(epoch, self.cfg.n_iters))
            print('-' * 10)
            self.scheduler.step()
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for I, (input_images, target_masks) in enumerate(data_iter):     #da I batch AL NUMERO DI BATCH PRESENTI
                inputs = input_images.to(self.device)
                labels = target_masks.to(self.device)
                outputs = self.model(inputs)
                self.reset_grad()   # resettiamo i  gradienti
                loss = self.c_loss(outputs, labels)
                loss.backward()
                self.optim.step()
                if I % 70 == 0:
                    # statistics
                    curr_loss = loss.item()
                    running_loss += curr_loss # average, DO NOT multiply by the batch size
                    output_label = torch.argmax(self.softmax(outputs), dim=1) #argmax
                    running_corrects += torch.sum(output_label == labels)
                    tv.utils.save_image(to_rgb(output_label.cpu()),f"result_{epoch}_{I}.jpg")  #f"pippo_{epoch}_{I}.jpg"
                    seconds = time.time() - since        #secondi sono uguali al tempo trascorso meno quello di training, cioe' quanto tempo ci ha messo a fare il training
                    elapsed = str(timedelta(seconds=seconds))
                    print('Iteration : [{iter}/{iters}]\t'
                                'minibatch: [{i}/{minibatch}]\t'
                                'Time : {time}\t'
                                'Running Correct : {corr}\t'
                                'Loss : {loss:.4f}\t'.format(i=I, minibatch=iters_per_epoch,
                                iter=epoch, iters=self.cfg.n_iters,
                                time=elapsed, corr=running_corrects, loss=curr_loss))

            if (epoch + 1) % self.cfg.log_step == 0:
                seconds = time.time() - since        #secondi sono uguali al tempo trascorso meno quello di training, cioe' quanto tempo ci ha messo a fare il training
                elapsed = str(timedelta(seconds=seconds))
                print('Iteration : [{iter}/{iters}]\t'
                    'Time : {time}\t'
                    'Running Correct : {corr}\t'
                    'Mean Loss Epoch : {mean_loss}\t'
                    'Total Loss Epoch: {loss:.4f}\t'.format(
                    iter=epoch, iters=self.cfg.n_iters,
                    time=elapsed, corr=running_corrects,
                    mean_loss=running_loss / iters_per_epoch,
                    loss=running_loss))




            self.save_network(self.model, "UNET_VOC", "latest", [0], epoch, self.optim, self.scheduler)
            if epoch % 10 == 0:
                self.save_network(self.model, "UNET_VOC", f"{epoch}", [0], epoch,
                                  self.optim, self.scheduler)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
