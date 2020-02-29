import torch
from torch import optim
import torch.nn as nn
import torchvision as tv
from torchvision import transforms
from datasets.voc import to_rgb
from models import unet
import torch.backends.cudnn as cudnn
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
        self.old_lr = self.cfg.lr
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
                     epoch, optimizer):


        if self.cfg.step == 'split_1':
            path = self.cfg.model_save_path+"/models_split1"
        elif self.cfg.step == 'split_2':
            path = self.cfg.model_save_path+"/models_split2"
        elif self.cfg.step == 'default':
            path = self.cfg.model_save_path+"/model_default"
                        
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)                  #salva la epoca ed il tipo di rete ( UNET) 
        save_path = os.path.join(path, save_filename)               #il path dove viene salvato
        print(save_path)                                                                #a schermo
        state = {
            "epoch": epoch + 1,
            "model_state": network.cpu().state_dict(),                                  #passa un dizionario che contiene lo stato e tutti i parametri
            "optimizer_state": optimizer.state_dict()
        }
        torch.save(state, save_path)                                                    #salviamo lo stato nel path
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda()

    ########### helper loading function that can be used by subclasses ###########
    def load_network(self, network, network_label, epoch_label, load_optim):
        
        

        if self.cfg.step == 'split_1':
            path = self.cfg.model_save_path+"/models_split1"
        elif self.cfg.step == 'split_2':
            path = self.cfg.model_save_path+"/models_split2"
        elif self.cfg.step == 'default':
            path = self.cfg.model_save_path+"/model_default"


        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not load_optim:
            path = self.cfg.model_save_path+"/models_split1"

        save_dir = path
        save_path = os.path.join(save_dir, save_filename)

        if not os.path.isfile(save_path) and self.cfg.step == 'split_2' and load_optim == True:
            path = self.cfg.model_save_path + "/models_split1"
            save_dir = path
            save_path = os.path.join(save_dir, save_filename)
            load_optim = False

        if not os.path.isfile(save_path):                                               #se non si trova nel path
            print('%s not exists yet!' % save_path)                                     #diciamo che non esiste! 
            if network_label == 'G':
                raise ('Generator must exist!')
        else:
            try:
                checkpoint = torch.load(save_path)                                      #checkpoint sarebbe una struttura ( tipo struct ?? )
                network.load_state_dict(checkpoint["model_state"])                      #gli passiamo lo stato con i parametri
                if load_optim:
                    self.start_epoch = checkpoint["epoch"]
                    self.optim.load_state_dict(checkpoint["optimizer_state"])
                print("Load model Done!")
            except:
                print("Error during the load of the model")                             #non viene importato

    ########### model builder ###########
    def build_model(self):
        
        self.model = unet.UNetWithResnet50Encoder(num_classes=21)
        self.optim = optim.Adam(self.model.parameters(),
                                lr=self.cfg.lr,
                                betas=[self.cfg.beta1, self.cfg.beta2])
        self.c_loss = nn.CrossEntropyLoss()
        self.distillation_loss = nn.KLDivLoss(reduction="none")
        self.softmax = nn.Softmax(dim=1).to(self.device)
        self.log_softmax = nn.LogSoftmax(dim=1).to(self.device)

        self.n_gpu = torch.cuda.device_count()
        if self.cfg.continue_train:
            self.load_network(self.model, "UNET_VOC", self.cfg.which_epoch, True)
        if self.n_gpu > 0:
            print('Use data parallel model(# gpu: {})'.format(self.n_gpu))
            self.model.cuda()
            self.model = nn.DataParallel(self.model)

        if self.n_gpu > 0:
            torch.backends.cudnn.benchmark = True
            for state in self.optim.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if self.cfg.step == "split_2":
            self.mask =\
                torch.from_numpy(
                    np.array(
                        [True if i < self.cfg.from_new_class else False for i in range(0,21)] * self.cfg.train_batch_size * 1 * self.cfg.h_image_size * self.cfg.w_image_size)).view(self.cfg.train_batch_size, -1, self.cfg.h_image_size, self.cfg.w_image_size)

            self.old_model = unet.UNetWithResnet50Encoder(num_classes=21)

            self.load_network(self.old_model, "UNET_VOC", self.cfg.which_epoch, False)
            if self.n_gpu > 0:
                print('Use data parallel model(# gpu: {})'.format(self.n_gpu))
                self.old_model.cuda()
                self.old_model = nn.DataParallel(self.old_model)
                for param in self.old_model.parameters():
                    param.requires_grad = False


    ########### trainer phase ###########
    def train_val(self):

        since = time.time()
        iters_per_epoch = len(self.train_data_loader.dataset) // self.cfg.train_batch_size
        epoch = self.start_epoch
        print(self.start_epoch)

        print(f"batch size {self.cfg.train_batch_size} dataset size : [{len(self.train_data_loader.dataset)}]"
              f" epoch : [{self.cfg.n_iters + self.cfg.n_iters_decay}]"
              f" iterations per epoch: {iters_per_epoch}")

        ########### train until model is fully trained  ###########
        while epoch < self.cfg.n_iters + self.cfg.n_iters_decay:
            print('Epoch {}/{}'.format(epoch, self.cfg.n_iters + self.cfg.n_iters_decay))
            print('-' * 10)
            running_loss = 0.0
            pixel_accuracy_epoch=0.0
            start_epoch = time.time()
            print_number = 0


            for I, (input_images, target_masks) in enumerate(iter(self.train_data_loader)):   

                start_mini_batch = time.time()
 

                inputs = input_images.to(self.device)
                labels = target_masks.to(self.device)  #, dtype=torch.int64
                

                outputs = self.model(inputs)     
                self.reset_grad()
                loss = self.c_loss(outputs, labels)
                loss_distillation = 0
                if self.cfg.lambda_distillation != 0.0:
                    outputs_old = self.old_model(inputs)
                    loss_distillation = self.distillation_loss(self.log_softmax(outputs), self.softmax(outputs_old))
                    loss_distillation = loss_distillation[self.mask]
                    loss_distillation = loss_distillation.mean()


                loss = loss + (self.cfg.lambda_distillation * loss_distillation)

                f = open('loss.csv', 'a')
                f.write(str(loss.item())+"\n")
                f.close()


                loss.backward()


                self.optim.step()
                if I % 200 == 0:
                    print_number += 1
                    acc_meter_mb = AverageMeter()
                    intersection_meter_mb = AverageMeter()
                    union_meter_mb = AverageMeter()
                    class_acc_meter_mb = AverageMeter()
                    class_acc_meter_mb.initialize(0,0, 21)
                    ########### statistics  ###########
                    curr_loss = loss.item()                                         #ritorna il valore del tensore
                    running_loss += curr_loss                                       #average, DO NOT multiply by the batch size
                    output_label = torch.argmax(self.softmax(outputs), dim=1)       #argmax
                    iou_mean = mt.intersectionAndUnion_torch(output_label, labels)
                    output_label = output_label.cpu()
                    labels = labels.cpu()
                    acc, pix = mt.accuracy(output_label, labels)
                    intersection, union =\
                        mt.intersectionAndUnion(output_label, labels,
                                                               21)
                    acc_meter_mb.update(acc, pix)
                    intersection_meter_mb.update(intersection)
                    union_meter_mb.update(union)
                    confusion_matrix = mt.class_accuracy(output_label,
                                                         labels,
                                                         class_acc_meter_mb.get_confusion_matrix(),
                                                         labels=range(0,21))
                    ########### printing out the model ###########

                    path=self.cfg.sample_save_path    
                    if self.cfg.step == 'split_1':
                        path=self.cfg.sample_save_path +"/samples_split1_training"
                    elif self.cfg.step == 'split_2':
                        path=self.cfg.sample_save_path +"/samples_split2_training"
                    elif self.cfg.step == 'default':
                        path=self.cfg.sample_save_path +"/samples_default_training"

                    tv.utils.save_image(to_rgb(output_label),os.path.join(path,"generated",f"predicted_{epoch}_{I}.jpg"), padding=100, normalize=True, range=(0,255))
                    tv.utils.save_image(to_rgb(labels),os.path.join(path,"ground_truth",f"ground_truth_{epoch}_{I}.jpg"), padding=100, normalize=True, range=(0,255))
                    tv.utils.save_image(inputs.cpu(),os.path.join(path,"inputs",f"input_{epoch}_{I}.jpg"),normalize=True, range=(-1,1), padding=100)

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
                                iter=epoch, iters=self.cfg.n_iters + self.cfg.n_iters_decay, mean=iou_mean,
                                time=elapsed, loss=curr_loss))
            if epoch > self.cfg.n_iters:
                if self.n_gpu > 0:
                    self.update_learning_rate()
                else:
                    self.update_learning_rate()
            ########### one epoch done  ###########                   
            if (epoch + 1) % self.cfg.log_step == 0:
                seconds = time.time() - start_epoch        
                elapsed = str(timedelta(seconds=seconds))
                seconds_from_beginning = time.time() - since
                elapsed_start = str(timedelta(seconds=seconds_from_beginning))
                print('Iteration : [{iter}/{iters}]\t'
                    'Epoch Time : {time_epoch}\t'
                    'Total Time : {time_start}\t'.format(
                    iter=epoch, iters=self.cfg.n_iters,
                    time_epoch=elapsed, time_start=elapsed_start))

            ########### eval phase  ###########
            if (epoch + 1) % self.cfg.test_step == 0:
                print("TESTING")
                test_acc, iou, confusion_matrix, iou_mean = self.test()
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
                    'Test Mean IOU : {mean_iou:.4f}\t'
                    'Test Mean Class Accuracy : {mean_ca:.4f}\t'
                    'Loss Epoch: {loss:.4f}\t'.format(
                    iter=epoch, iters=self.cfg.n_iters + self.cfg.n_iters_decay,
                    test = test_acc * 100,
                    time_epoch=elapsed, time_start=elapsed_start,
                    mean_iou=iou_mean,
                    mean_ca=classes_acc.mean(),
                    loss=running_loss / print_number))
                print("FINE TESTING")
            epoch +=1



            self.save_network(self.model.module, "UNET_VOC", "latest", [0], epoch, self.optim)
            if epoch % 10 == 0:
                self.save_network(self.model.module, "UNET_VOC", f"{epoch}", [0], epoch,
                                  self.optim)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))



    ########### eval phase ###########
    def test(self):
        acc_meter_test = AverageMeter()
        intersection_meter_test = AverageMeter()
        union_meter_test = AverageMeter()
        class_acc_meter_test = AverageMeter()
        class_acc_meter_test.initialize(0,0, 21)
        iou_mean = 0
        with torch.no_grad():
            self.model.eval()
            for i, (images, labels) in enumerate(self.val_data_loader):
                if torch.cuda.is_available():
                    images = Variable(images.cuda())
                    labels = Variable(labels.cuda())
                outputs = self.model(images)
                prediction = torch.argmax(self.softmax(outputs), dim=1)
                prediction = prediction.cpu()
                labels = labels.cpu()
                acc, pix = mt.accuracy(prediction, labels)
                intersection, union = \
                    mt.intersectionAndUnion(prediction, labels,
                                            21)
                acc_meter_test.update(acc, pix)
                intersection_meter_test.update(intersection)
                union_meter_test.update(union)
                confusion_matrix_epoch = mt.class_accuracy(
                    prediction,
                    labels,
                    class_acc_meter_test.get_confusion_matrix(),
                    labels=range(0,21))
                class_acc_meter_test.update_confusion_matrix(
                    confusion_matrix_epoch)
                path = self.cfg.sample_save_path
                if self.cfg.step == 'split_1':
                    path = self.cfg.sample_save_path + "/samples_split1_testing"
                elif self.cfg.step == 'split_2':
                    path = self.cfg.sample_save_path + "/samples_split2_testing"
                elif self.cfg.step == 'default':
                    path = self.cfg.sample_save_path + "/samples_default_testing"

                tv.utils.save_image(to_rgb(prediction),
                                    os.path.join(path, "generated",
                                                 f"predicted_testing_{i}.jpg"),
                                    padding=100, normalize=True, range=(0,255))
                tv.utils.save_image(to_rgb(labels),
                                    os.path.join(path, "ground_truth",
                                                 f"ground_truth_testing_{i}.jpg"),
                                    padding=100, normalize=True, range=(0,255))
                tv.utils.save_image(images.cpu(), os.path.join(path, "inputs",
                                                               f"input_testing_{i}.jpg"),
                                    normalize=True, range=(-1, 1), padding=100)

        iou = intersection_meter_test.sum / (union_meter_test.sum + 1e-10)
        return acc_meter_test.average(), iou, class_acc_meter_test.confusion_matrix, np.mean(iou)

    def update_learning_rate(self):
        lrd = self.cfg.lr / self.cfg.n_iters_decay
        lr = self.old_lr - lrd
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr
        self.old_lr = lr

    def print_results(self, results):
        message = '(test epoch)'
        for k, v in results.items():
            if v != 0:
                message += '%s: %.7f ' % (k, v)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)