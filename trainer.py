import torch
from torch import optim
import torch.nn as nn
import torchvision as tv
from torchvision import transforms
from models import unet
from datasets.voc import to_rgb
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
from utils import AverageMeter
import numpy as np
import os, sys
import time
from datetime import timedelta
from torch.autograd import Variable
import metrics as mt

try:
    import nsml
    from nsml import Visdom
    USE_NSML = True
    print('NSML imported')
except ImportError:
    print('Cannot Import NSML. Use local GPU')
    USE_NSML = False

cudnn.benchmark = True                                                                  # For fast speed

def imshow(self, img):
    img = img / 2 + 0.5                                                                 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


########### Initialization ###########

class Trainer:
    def __init__(self, train_data_loader, val_data_loader, config):                     #gli passiamo i due dataset e la stringa di comandi
        self.cfg = config
        self.train_data_loader = train_data_loader                                      #associa il dataset train
        self.val_data_loader = val_data_loader                                          #associa il dataset val
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_epoch = 0
        self.build_model()                                                              #crea il modello

    def imshow(self, img):
        print("entrato")
        img = img / 2 + 0.5                                                             # unnormalize
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
        return out.clamp_(0, 1)                                                         #ritorna il valore, 0 se minore di 0 e 1 se maggiore di 1. lo " NORMALIZZA "

    def reset_grad(self):
        self.optim.zero_grad()                                                          #resetta il gradiente

    
    ########### helper saving function that can be used by subclasses ###########
    def save_network(self, network, network_label, epoch_label, gpu_ids,
                     epoch, optimizer, scheduler):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)                  #salva la epoca ed il tipo di rete ( UNET) 
        save_path = os.path.join(self.cfg.model_save_path, save_filename)               #il path dove viene salvato
        print(save_path)                                                                #a schermo
        state = {
            "epoch": epoch + 1,
            "model_state": network.cpu().state_dict(),                                  #passa un dizionario che contiene lo stato e tutti i parametri
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict()
        }
        torch.save(state, save_path)                                                    #salviamo lo stato nel path
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda()

    ########### helper loading function that can be used by subclasses ###########
    def load_network(self, network, network_label, epoch_label,
                     epoch, optimizer, scheduler, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)  
        save_dir = self.cfg.model_save_path
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):                                               #se non si trova nel path
            print('%s not exists yet!' % save_path)                                     #diciamo che non esiste! 
            if network_label == 'G':
                raise ('Generator must exist!')
        else:
            try:
                checkpoint = torch.load(save_path)                                      #checkpoint sarebbe una struttura ( tipo struct ?? )
                network.load_state_dict(checkpoint["model_state"])                      #gli passiamo lo stato con i parametri
                self.start_epoch = checkpoint["epoch"]
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                scheduler.load_state_dict(checkpoint["scheduler_state"])
                print("Load model Done!")
            except:
                print("Error during the load of the model")                             #non viene importato

    ########### model builder ###########
    def build_model(self):
        
        self.model = unet.UNet(num_classes=21, in_dim=3, conv_dim=64)
        self.optim = optim.Adam(self.model.parameters(),                                #usiamo adam per ottimizzazione stocastica come OPTIM, passangogli i parametri
                                lr=self.cfg.lr,                                         #settiamo il learning rate
                                betas=[self.cfg.beta1, self.cfg.beta2])                 #le due Beta, cioe' la probabilita' di accettare l'ipotesi quando e' falsa  (coefficients used for computing running averages of gradient and its square )
        lr_lambda = lambda n_iter: (1 - n_iter/self.cfg.n_iters)**self.cfg.lr_exp       #ATTENZIONE: learning rate LAMBDA penso
        self.scheduler = LambdaLR(self.optim, lr_lambda=lr_lambda)
        self.c_loss = nn.CrossEntropyLoss().to(self.device)                             #crossEntropy ! muove il modello nella GPU
        self.softmax = nn.Softmax(dim=1).to(self.device)                                # channel-wise softmax             #facciamo il softmax, cioe' prendiamo tutte le probabilita' e facciamo in modo che la loro somma sia 1

        self.n_gpu = torch.cuda.device_count()                                          #ritorna il numero di GPU a disposizione
        if self.cfg.continue_train:
            self.load_network(self.model, "UNET_VOC", self.cfg.which_epoch,
                              self.start_epoch, self.optim, self.scheduler)
        if self.n_gpu > 1:
            print('Use data parallel model(# gpu: {})'.format(self.n_gpu))
            self.model = nn.DataParallel(self.model)                                    #implementa il parallelismo, se disponibile
        self.model = self.model.to(self.device)
        if self.n_gpu > 0:
            torch.backends.cudnn.benchmark = True
            for state in self.optim.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

    def unisci_tensori(self,train_data_1_imag,train_data_1_mask):
        
        
        list_data_1 = []
        list_data_2 = []
        list_data_3 = []
        
        train_data_1 = []
        train_data_2 = []
        cont =0
        print(len(train_data_1_imag))
        while( cont < len(train_data_1_imag) ):
            
            list_data_1.append(train_data_1_mask[cont])
            list_data_2.append(train_data_1_imag[cont])

            if ((cont%15==0 and cont != 0) or cont == (len(train_data_1_imag)-1) ):
                
                ####  questo pezzo di codice e' un abominio. purtroppo pytorch non permette di usare stack su cicli per concatenare tensori :(   che vergogna) ####
                if(len(list_data_1)==1):
                    train_data_1 = torch.stack([list_data_1[0][0]])
                    train_data_2 = torch.stack([list_data_2[0][0]])
                elif(len(list_data_1)==2):
                    train_data_1 = torch.stack([list_data_1[0][0],list_data_1[1][0]])
                    train_data_2 = torch.stack([list_data_2[0][0],list_data_2[1][0]])
                elif(len(list_data_1)==3):
                    train_data_1 = torch.stack([list_data_1[0][0],list_data_1[1][0],list_data_1[2][0]])
                    train_data_2 = torch.stack([list_data_2[0][0],list_data_2[1][0],list_data_2[2][0]])
                elif(len(list_data_1)==4):
                    train_data_1 = torch.stack([list_data_1[0][0],list_data_1[1][0],list_data_1[2][0],list_data_1[3][0]])
                    train_data_2 = torch.stack([list_data_2[0][0],list_data_2[1][0],list_data_2[2][0],list_data_2[3][0]])

                elif(len(list_data_1)==5):
                    train_data_1 = torch.stack([list_data_1[0][0],list_data_1[1][0],list_data_1[2][0],list_data_1[3][0],list_data_1[4][0]])
                    train_data_2 = torch.stack([list_data_2[0][0],list_data_2[1][0],list_data_2[2][0],list_data_2[3][0],list_data_2[4][0]])

                elif(len(list_data_1)==6):
                    train_data_1 = torch.stack([list_data_1[0][0],list_data_1[1][0],list_data_1[2][0],list_data_1[3][0],list_data_1[4][0],list_data_1[5][0]])
                    train_data_2 = torch.stack([list_data_2[0][0],list_data_2[1][0],list_data_2[2][0],list_data_2[3][0],list_data_2[4][0],list_data_2[5][0]])

                elif(len(list_data_1)==7):
                    train_data_1 = torch.stack([list_data_1[0][0],list_data_1[1][0],list_data_1[2][0],list_data_1[3][0],list_data_1[4][0],list_data_1[5][0],list_data_1[6][0]])
                    train_data_2 = torch.stack([list_data_2[0][0],list_data_2[1][0],list_data_2[2][0],list_data_2[3][0],list_data_2[4][0],list_data_2[5][0],list_data_2[6][0]])

                elif(len(list_data_1)==8):
                    train_data_1 = torch.stack([list_data_1[0][0],list_data_1[1][0],list_data_1[2][0],list_data_1[3][0],list_data_1[4][0],list_data_1[5][0],list_data_1[6][0],list_data_1[7][0]])
                    train_data_2 = torch.stack([list_data_2[0][0],list_data_2[1][0],list_data_2[2][0],list_data_2[3][0],list_data_2[4][0],list_data_2[5][0],list_data_2[6][0],list_data_2[7][0]])

                elif(len(list_data_1)==9):
                    train_data_1 = torch.stack([list_data_1[0][0],list_data_1[1][0],list_data_1[2][0],list_data_1[3][0],list_data_1[4][0],list_data_1[5][0],list_data_1[6][0],list_data_1[7][0],list_data_1[8][0]])
                    train_data_2 = torch.stack([list_data_2[0][0],list_data_2[1][0],list_data_2[2][0],list_data_2[3][0],list_data_2[4][0],list_data_2[5][0],list_data_2[6][0],list_data_2[7][0],list_data_2[8][0]])

                elif(len(list_data_1)==10):
                    train_data_1 = torch.stack([list_data_1[0][0],list_data_1[1][0],list_data_1[2][0],list_data_1[3][0],list_data_1[4][0],list_data_1[5][0],list_data_1[6][0],list_data_1[7][0],list_data_1[8][0],list_data_1[9][0]])
                    train_data_2 = torch.stack([list_data_2[0][0],list_data_2[1][0],list_data_2[2][0],list_data_2[3][0],list_data_2[4][0],list_data_2[5][0],list_data_2[6][0],list_data_2[7][0],list_data_2[8][0],list_data_2[9][0]])

                elif(len(list_data_1)==11):
                    train_data_1 = torch.stack([list_data_1[0][0],list_data_1[1][0],list_data_1[2][0],list_data_1[3][0],list_data_1[4][0],list_data_1[5][0],list_data_1[6][0],list_data_1[7][0],list_data_1[8][0],list_data_1[9][0],list_data_1[10][0]])
                    train_data_2 = torch.stack([list_data_2[0][0],list_data_2[1][0],list_data_2[2][0],list_data_2[3][0],list_data_2[4][0],list_data_2[5][0],list_data_2[6][0],list_data_2[7][0],list_data_2[8][0],list_data_2[9][0],list_data_2[10][0]])

                elif(len(list_data_1)==12):
                    train_data_1 = torch.stack([list_data_1[0][0],list_data_1[1][0],list_data_1[2][0],list_data_1[3][0],list_data_1[4][0],list_data_1[5][0],list_data_1[6][0],list_data_1[7][0],list_data_1[8][0],list_data_1[9][0],list_data_1[10][0],list_data_1[11][0]])
                    train_data_2 = torch.stack([list_data_2[0][0],list_data_2[1][0],list_data_2[2][0],list_data_2[3][0],list_data_2[4][0],list_data_2[5][0],list_data_2[6][0],list_data_2[7][0],list_data_2[8][0],list_data_2[9][0],list_data_2[10][0],list_data_2[11][0]])

                elif(len(list_data_1)==13):
                    train_data_1 = torch.stack([list_data_1[0][0],list_data_1[1][0],list_data_1[2][0],list_data_1[3][0],list_data_1[4][0],list_data_1[5][0],list_data_1[6][0],list_data_1[7][0],list_data_1[8][0],list_data_1[9][0],list_data_1[10][0],list_data_1[11][0],list_data_1[12][0]])
                    train_data_2 = torch.stack([list_data_2[0][0],list_data_2[1][0],list_data_2[2][0],list_data_2[3][0],list_data_2[4][0],list_data_2[5][0],list_data_2[6][0],list_data_2[7][0],list_data_2[8][0],list_data_2[9][0],list_data_2[10][0],list_data_2[11][0],list_data_2[12][0]])

                elif(len(list_data_1)==14):
                    train_data_1 = torch.stack([list_data_1[0][0],list_data_1[1][0],list_data_1[2][0],list_data_1[3][0],list_data_1[4][0],list_data_1[5][0],list_data_1[6][0],list_data_1[7][0],list_data_1[8][0],list_data_1[9][0],list_data_1[10][0],list_data_1[11][0],list_data_1[12][0],list_data_1[13][0]])
                    train_data_2 = torch.stack([list_data_2[0][0],list_data_2[1][0],list_data_2[2][0],list_data_2[3][0],list_data_2[4][0],list_data_2[5][0],list_data_2[6][0],list_data_2[7][0],list_data_2[8][0],list_data_2[9][0],list_data_2[10][0],list_data_2[11][0],list_data_2[12][0],list_data_2[13][0]])

                elif(len(list_data_1)==15):
                    train_data_1 = torch.stack([list_data_1[0][0],list_data_1[1][0],list_data_1[2][0],list_data_1[3][0],list_data_1[4][0],list_data_1[5][0],list_data_1[6][0],list_data_1[7][0],list_data_1[8][0],list_data_1[9][0],list_data_1[10][0],list_data_1[11][0],list_data_1[12][0],list_data_1[13][0],list_data_1[14][0]])
                    train_data_2 = torch.stack([list_data_2[0][0],list_data_2[1][0],list_data_2[2][0],list_data_2[3][0],list_data_2[4][0],list_data_2[5][0],list_data_2[6][0],list_data_2[7][0],list_data_2[8][0],list_data_2[9][0],list_data_2[10][0],list_data_2[11][0],list_data_2[12][0],list_data_2[13][0],list_data_2[14][0]])

                elif(len(list_data_1)==16):
                    train_data_1 = torch.stack([list_data_1[0][0],list_data_1[1][0],list_data_1[2][0],list_data_1[3][0],list_data_1[4][0],list_data_1[5][0],list_data_1[6][0],list_data_1[7][0],list_data_1[8][0],list_data_1[9][0],list_data_1[10][0],list_data_1[11][0],list_data_1[12][0],list_data_1[13][0],list_data_1[14][0],list_data_1[15][0]])
                    train_data_2 = torch.stack([list_data_2[0][0],list_data_2[1][0],list_data_2[2][0],list_data_2[3][0],list_data_2[4][0],list_data_2[5][0],list_data_2[6][0],list_data_2[7][0],list_data_2[8][0],list_data_2[9][0],list_data_2[10][0],list_data_2[11][0],list_data_2[12][0],list_data_2[13][0],list_data_2[14][0],list_data_2[15][0]])


                   
                
                
                
                list_data_3.append([train_data_2,train_data_1])
                list_data_1.clear()
                list_data_2.clear() 

            cont = cont+1 
        
        return list_data_3


    ########### trainer phase ###########
    def train_val(self):
        since = time.time()
        iters_per_epoch = len(self.train_data_loader.dataset) // self.cfg.train_batch_size
        epoch = self.start_epoch
        print(self.start_epoch)

        print(f"batch size {self.cfg.train_batch_size} dataset size : [{len(self.train_data_loader.dataset)}]"
              f" epoch : [{self.cfg.n_iters}]"
              f" iterations per epoch: {iters_per_epoch}")

        ########### train until model is fully trained  ###########
        while epoch < self.cfg.n_iters:
            acc_meter_epoch = AverageMeter()
            intersection_meter_epoch = AverageMeter()
            union_meter_epoch = AverageMeter()
            class_acc_meter_epoch = AverageMeter()
            class_acc_meter_epoch.initialize(0, 0, 22)
            print('Epoch {}/{}'.format(epoch, self.cfg.n_iters))
            print('-' * 10)
            self.scheduler.step()
            running_loss = 0.0
            pixel_accuracy_epoch=0.0
            start_epoch = time.time()
            print_number = 0



            train_data_1 = []
            train_data_2 = []
            train_data_3 = []

            train_data_1_mask = []
            train_data_2_mask = []
            train_data_3_mask = []


            train_data_1_imag = []
            train_data_2_imag = []
            train_data_3_imag = []

            mascheraModificata = Variable(torch.randn(self.cfg.train_batch_size, self.cfg.h_image_size, self.cfg.w_image_size))  #inizializzo un tensore che mi servira' per copiare un altro tensore mask, senza le classi diverse da quella piu' presente
            print(mascheraModificata.size())
            for i, (image, mask) in enumerate(iter(self.train_data_loader)):
                    
                    #mask:   numero di immagini + altezza + larghezza
                    #image:   numero di immagini + numero di channels + altezza + larghezza

                for I in range(self.cfg.train_batch_size):
                    out = mask[I].numpy().flatten()   # la matrice diventa un vettore
                    try:    
                            b = np.argmax(np.bincount(out[out != (0 or 21)]))   # ritorno il valore B maggior presente del vettore, diverso dallo sfondo (0) o void (21)                                                             print("i:",i," ,I: ", I," B:" ,b)
                            mascheraModificata[I] = torch.from_numpy(np.where(out!=(b and 0), 21, out).reshape( self.cfg.h_image_size, self.cfg.w_image_size)) #sostituisco tutti gli altri valori diversi da B o background, con un valore nullo                        mascheraModificata = torch.from_numpy(newReplaced)
                            print(i, " fatto")
                    except:
                            mascheraModificata[I] = mask[I] 
                            print(i, " An exception occurred")
                    if (b<12):
                            train_data_1_mask.append([mascheraModificata[I]])
                            train_data_1_imag.append([image[I]])
                    elif(b<17):
                            train_data_2_mask.append([mascheraModificata[I]])   
                            train_data_2_imag.append([image[I]])
                    else:
                            train_data_3_mask.append([mascheraModificata[I]])
                            train_data_3_imag.append([image[I]])
            

            train_data_1 = self.unisci_tensori(train_data_1_imag,train_data_1_mask)
            train_data_2 = self.unisci_tensori(train_data_2_imag,train_data_2_mask)
            train_data_3 = self.unisci_tensori(train_data_3_imag,train_data_3_mask)
            
            
            
            print(len(train_data_1))
            print(len(train_data_2))
            print(len(train_data_3))


            ########### Iterate over data ###########
            #for I, (input_images, target_masks) in enumerate(iter(self.train_data_loader)):   
            for I, (input_images, target_masks) in enumerate(train_data_1):    
            
                #for I in range(len(train_data_1)):    
            
                #input_images = train_data_1[I][0]
                #target_masks = train_data_1[I][1]

                print("input",input_images.size())
                print(type(input_images))
                print(type(target_masks))
                print("target",target_masks.size())

                start_mini_batch = time.time()
 

                inputs = input_images.to(self.device)
                labels = target_masks.to(self.device, dtype=torch.int64) 
                

                outputs = self.model(inputs)     
                self.reset_grad()                                                   #resettiamo i  gradienti
                loss = self.c_loss(outputs, labels)                                 #cross entropy tra l'output e quello che avremmo dovuto ottenere
                loss.backward()                                                     #fa il gradiente
                self.optim.step()                                                   #ottimizza tramite adam
                if I % 20 == 0:
                    print_number += 1
                    acc_meter_mb = AverageMeter()
                    intersection_meter_mb = AverageMeter()
                    union_meter_mb = AverageMeter()
                    class_acc_meter_mb = AverageMeter()
                    class_acc_meter_mb.initialize(0,0, 22)
                    ########### statistics  ###########
                    curr_loss = loss.item()                                         #ritorna il valore del tensore 
                    running_loss += curr_loss                                       #average, DO NOT multiply by the batch size
                    output_label = torch.argmax(self.softmax(outputs), dim=1)       #argmax
                    
                    acc, pix = mt.accuracy(output_label.cpu(), labels.cpu())
                    intersection, union =\
                        mt.intersectionAndUnion(output_label.cpu(), labels.cpu(),
                                                               22)
                    acc_meter_mb.update(acc, pix)
                    intersection_meter_mb.update(intersection)
                    union_meter_mb.update(union)
                    confusion_matrix = mt.class_accuracy(output_label.cpu(),
                                                         labels.cpu(),
                                                         class_acc_meter_mb.get_confusion_matrix())
                    confusion_matrix_epoch = mt.class_accuracy(output_label.cpu(),
                                                         labels.cpu(),
                                                         class_acc_meter_epoch.get_confusion_matrix())
                    acc_meter_epoch.update(acc, pix)
                    intersection_meter_epoch.update(intersection)
                    union_meter_epoch.update(union)
                    class_acc_meter_epoch.update_confusion_matrix(confusion_matrix_epoch)
                    ########### printing out the model ###########
                    tv.utils.save_image(to_rgb(output_label.cpu()),os.path.join(self.cfg.sample_save_path,"generated",f"predicted_{epoch}_{I}.jpg")) 
                    tv.utils.save_image(to_rgb(labels.cpu()),os.path.join(self.cfg.sample_save_path,"ground_truth",f"ground_truth_{epoch}_{I}.jpg"))  
                    tv.utils.save_image(inputs.cpu(),os.path.join(self.cfg.sample_save_path,"inputs",f"input_{epoch}_{I}.jpg"),normalize=True, range=(-1,1))  

                    seconds = time.time() - start_mini_batch        
                    elapsed = str(timedelta(seconds=seconds))
                    iou = intersection_meter_mb.sum / (union_meter_mb.sum + 1e-10)
                    classes_acc = confusion_matrix.diag()/(confusion_matrix.sum(1)+ 1e-10) * 100
                    for i, class_acc in enumerate(classes_acc):
                        print('class [{}], Mean acc: {:.4f}'.format(i, class_acc))
                    for i, _iou in enumerate(iou):
                        print('class [{}], IoU: {:.4f}'.format(i, _iou))
                    print('Iteration : [{iter}/{iters}]\t'
                                'minibatch: [{i}/{minibatch}]\t'
                                'Mini Batch Time : {time}\t'
                                'Pixel Accuracy : {acc:.4f}%\t'
                                'Mean IOU : {mean:.4f}\t'
                                'Mini Batch Loss : {loss:.4f}\t'.format(i=I, minibatch=iters_per_epoch,
                                acc = acc_meter_mb.average()*100,
                                iter=epoch, iters=self.cfg.n_iters, mean=iou.mean(),
                                time=elapsed, loss=curr_loss))
                else:
                    output_label = torch.argmax(self.softmax(outputs),
                                                dim=1)  # argmax
                    acc, pix = mt.accuracy(output_label.cpu(), labels.cpu())
                    intersection, union = \
                        mt.intersectionAndUnion(output_label.cpu(),
                                                labels.cpu(),
                                                22)
                    acc_meter_epoch.update(acc, pix)
                    intersection_meter_epoch.update(intersection)
                    union_meter_epoch.update(union)
                    confusion_matrix_epoch = mt.class_accuracy(
                        output_label.cpu(),
                        labels.cpu(),
                        class_acc_meter_epoch.get_confusion_matrix())
                    class_acc_meter_epoch.update_confusion_matrix(
                        confusion_matrix_epoch)

            ########### one epoch done  ###########                   
            if (epoch + 1) % self.cfg.log_step == 0:
                seconds = time.time() - start_epoch        
                elapsed = str(timedelta(seconds=seconds))
                seconds_from_beginning = time.time() - since
                elapsed_start = str(timedelta(seconds=seconds_from_beginning))
                iou = intersection_meter_epoch.sum / (union_meter_epoch.sum + 1e-10)
                classes_acc = class_acc_meter_epoch.confusion_matrix.diag() / (
                            class_acc_meter_epoch.confusion_matrix.sum(1) + 1e-10) * 100
                for i, class_acc in enumerate(classes_acc):
                    print('class [{}], Mean acc: {:.4f}'.format(i, class_acc))
                for i, _iou in enumerate(iou):
                    print('class [{}], IoU: {:.4f}'.format(i, _iou))
                print('Iteration : [{iter}/{iters}]\t'
                    'Epoch Time : {time_epoch}\t'
                    'Total Time : {time_start}\t'
                    'Accuracy Epoch : {acc}\t'
                      'Mean IOU : {mean:.4f}\t'
                    'Loss Epoch: {loss:.4f}\t'.format(
                    iter=epoch, iters=self.cfg.n_iters,
                    time_epoch=elapsed, time_start=elapsed_start,
                    acc =acc_meter_epoch.average()*100,
                    mean=iou.mean(),
                    loss=running_loss / print_number))

            ########### eval phase  ###########
            if (epoch + 1) % 150 == 0:
                test_acc, iou, confusion_matrix = self.test()
                seconds = time.time() - start_epoch                                 #secondi sono uguali al tempo trascorso meno quello di training, cioe' quanto tempo ci ha messo a fare il training
                elapsed = str(timedelta(seconds=seconds))
                seconds_from_beginning = time.time() - since
                elapsed_start = str(timedelta(seconds=seconds_from_beginning))
                classes_acc = confusion_matrix.diag() / (
                            confusion_matrix.sum(1) + 1e-10) * 100
                for i, class_acc in enumerate(classes_acc):
                    print('class [{}], Mean acc: {:.4f}'.format(i, class_acc))
                for i, _iou in enumerate(iou):
                    print('class [{}], IoU: {:.4f}'.format(i, _iou))
                print('Iteration : [{iter}/{iters}]\t'
                    'Epoch Time : {time_epoch}\t'
                    'Total Time : {time_start}\t'
                    'Test Accuracy  : {test}\t'
                    'Loss Epoch: {loss:.4f}\t'.format(
                    iter=epoch, iters=self.cfg.n_iters,
                    test = test_acc * 100,
                    time_epoch=elapsed, time_start=elapsed_start,
                    loss=running_loss / print_number))
            epoch +=1



            self.save_network(self.model, "UNET_VOC", "latest", [0], epoch, self.optim, self.scheduler)         #salva l'ultima epoca
            if epoch % 10 == 0:
                self.save_network(self.model, "UNET_VOC", f"{epoch}", [0], epoch,
                                  self.optim, self.scheduler)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))



    ########### eval phase ###########
    def test(self):
        acc_meter_test = AverageMeter()
        intersection_meter_test = AverageMeter()
        union_meter_test = AverageMeter()
        class_acc_meter_test = AverageMeter()
        class_acc_meter_test.initialize(0,0, 22)

        self.model.eval()
        for i, (images, labels) in enumerate(self.val_data_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            outputs = self.model(images)
            _, prediction = torch.max(outputs.data, 1)
            acc, pix = mt.accuracy(prediction.cpu(), labels.cpu())
            intersection, union = \
                mt.intersectionAndUnion(prediction.cpu(), labels.cpu(),
                                        22)
            acc_meter_test.update(acc, pix)
            intersection_meter_test.update(intersection)
            union_meter_test.update(union)
            confusion_matrix_epoch = mt.class_accuracy(
                prediction.cpu(),
                labels.cpu(),
                class_acc_meter_test.get_confusion_matrix())
            class_acc_meter_test.update_confusion_matrix(
                confusion_matrix_epoch)

        iou = intersection_meter_test.sum / (union_meter_test.sum + 1e-10)
        return acc_meter_test.average(), iou, class_acc_meter_test.confusion_matrix
