import numpy as np
import torch


    #numero di pixel correttamente classificati / numero di erroreamente non pixel classificati ( falsi negativi)
def pixel_acc( mask, predicted, total_train, correct_train):
    total_train += mask.nelement()          #conta gli elementi all'interno del tensore
    train_accuracy = 100 * correct_train / total_train
    return train_accuracy, total_train, correct_train

#pixel accuracy * 1 / ( sommatoria ( falsi negativi - pixel correttamente classificati))  * 1 / numero di classi   ``#        """ Calculate mean Intersection over Union """
def mean_IU_(target, prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def pixel_accuracy(prediction, groud_truth):
  
  #  true_positives / pixel_totali_i #

    check_size(prediction, groud_truth)                          #controllo se le matrici hanno la stessa dimensione

    classi, numero_classi = extract_classes(groud_truth)                  #ritorna la dimensione di groud_truth, togliendo gli elementi comuni
    prediction_mask, groud_truth_mask = extract_both_masks(prediction, groud_truth, classi, numero_classi)     #ritorna array dove gli elementi sono uguali a quelli del prediction

    true_positives = 0
    pixel_totali_i  = 0

    for i, classe in enumerate(classi):
        curr_prediction_mask = prediction_mask[i, :, :]             #crea una copia del prediction mask
        curr_groud_truth_mask = groud_truth_mask[i, :, :]     #crea una copia del groud_truth mask

        true_positives += np.sum(np.logical_and(curr_prediction_mask, curr_groud_truth_mask))  
        pixel_totali_i  += np.sum(curr_groud_truth_mask)       #somma di tutti i pixel della groud_truth mask
 
    if (pixel_totali_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = ( true_positives * 100 ) / pixel_totali_i

    return pixel_accuracy_

def mean_accuracy(prediction, groud_truth):

    #   (true_positives / pixel_totali_i) / numero_di_groud_truth_diversi   #
    
    check_size(prediction, groud_truth)                                                                  #controllo se le matrici hanno la stessa dimensione

    classi, numero_classi = extract_classes(groud_truth)                                                 #estraggo groud_truth e tolgo i duplicati
    prediction_mask, groud_truth_mask = extract_both_masks(prediction, groud_truth, classi, numero_classi)        #ritorna array dove gli elementi sono uguali a quelli del prediction

    accuracy = list([0]) * numero_classi                                                            #e' una lista che e' lunga tanto il numero di classi

    for i, classe in enumerate(classi):                              
        curr_prediction_mask = prediction_mask[i, :, :]                                                     #creo copia di prediction mask
        curr_groud_truth_mask = groud_truth_mask[i, :, :]                                             #creo copia di groud_truth mask

        true_positives = np.sum(np.logical_and(curr_prediction_mask, curr_groud_truth_mask))               #come prima faccio la somma di tutti i punti che sono uguali
        pixel_totali_i  = np.sum(curr_groud_truth_mask)                                                #somma di tutti i pixel della prediction mask ( cioe' delle classi )                                     
 
        if (pixel_totali_i != 0):                                                       
            accuracy[i] = true_positives / pixel_totali_i                        

    mean_accuracy_ = np.mean(accuracy)                                                              #calcola la media della accuracy per tutta la lista 
    return mean_accuracy_

def mean_IoU(prediction, groud_truth):
    
    # (1/numero_classi) * sum_i(true_positives / (pixel_totali_i + sum_j(n_ji) - true_positives))# 
    

    check_size(prediction, groud_truth)                                                                  #controllo se le matrici hanno la stessa dimensione

    classi, numero_classi   = union_classes(prediction, groud_truth)
    _, numero_classi_ground_truth = extract_classes(groud_truth)
    prediction_mask, groud_truth_mask = extract_both_masks(prediction, groud_truth, classi, numero_classi)

    IoU = list([0]) * numero_classi

    for i, c in enumerate(classi):
        curr_prediction_mask = prediction_mask[i, :, :]                                                 #fa una copia del predicted
        curr_groud_truth_mask = groud_truth_mask[i, :, :]                                               #fa una copia del groud truth
 
        if (np.sum(curr_prediction_mask) == 0) or (np.sum(curr_groud_truth_mask) == 0):                 #se la somma e' 0, esce dal loop
            continue

        true_positives = np.sum(np.logical_and(curr_prediction_mask, curr_groud_truth_mask))              
        pixel_totali_i  = np.sum(curr_groud_truth_mask)
        false_positives = np.sum(curr_prediction_mask)

        IoU[i] = true_positives / (pixel_totali_i + false_positives - true_positives)
 
    mean_IoU_ = np.sum(IoU) / numero_classi_ground_truth                                                #fa la media per classe
    return mean_IoU_

def frequency_weighted_IU(prediction, groud_truth):
    
    # sum_k(t_k)^(-1) * sum_i((pixel_totali_i*true_positives)/(pixel_totali_i + sum_j(n_ji) - true_positives))# 

    check_size(prediction, groud_truth)                                                                 #controllo se le matrici hanno la stessa dimensione

    classi, numero_classi = union_classes(prediction, groud_truth)                                          #unisce le classi, confrontando gli elementi uguali
    prediction_mask, groud_truth_mask = extract_both_masks(prediction, groud_truth, classi, numero_classi)

    frequency_weighted_IU_ = list([0]) * numero_classi                                                  #lista lunga il numero di classi

    for i, classi in enumerate(cl):
        curr_prediction_mask = prediction_mask[i, :, :]                                                 #copia del prediction
        curr_groud_truth_mask = groud_truth_mask[i, :, :]                                               #copia del ground truth
 
        if (np.sum(curr_prediction_mask) == 0) or (np.sum(curr_groud_truth_mask) == 0):                 #se la somma dei pixel = 0, esce dal loop
            continue

        true_positives = np.sum(np.logical_and(curr_prediction_mask, curr_groud_truth_mask))              
        pixel_totali_i  = np.sum(curr_groud_truth_mask)
        false_positives = np.sum(curr_prediction_mask)

        frequency_weighted_IU_[i] = (pixel_totali_i * true_positives) / (pixel_totali_i + false_positives - true_positives)
 
    sum_k_t_k = get_pixel_area(prediction)
    
    frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k                                 #media per classe
    return frequency_weighted_IU_


#Funzioni ausiliari per evaluation #

def get_pixel_area(segmento):
    return segmento.shape[0] * segmento.shape[1]

def extract_both_masks(prediction, groud_truth, classi, numero_classi):
    prediction_mask = extract_masks(prediction, classi, numero_classi)          
    groud_truth_mask   = extract_masks(groud_truth, classi, numero_classi)

    return prediction_mask, groud_truth_mask

def extract_classes(segmento):
    classi = np.unique(segmento)                                         #elimina i doppioni
    numero_classi = len(classi)                                          #ritorna la lunghezza della mia tupla

    return classi, numero_classi                                         #ritorna la tupla senza doppioni e la sua dimensione( cioe' quante classi abbiamo)

def union_classes(prediction, groud_truth):                      
    prediction_cl, _ = extract_classes(prediction)                               #ritorna tupla senza doppioni di prediction
    groud_truth_cl, _ = extract_classes(groud_truth)                       #ritorna tupla senza doppioni di groud_truth

    classi = np.union1d(prediction_cl, groud_truth_cl)                        #ritorna l'unione dei 2 array
    numero_classi = len(classi)                                          #ritorna la lunghezza dell'unione

    return classi, numero_classi                                         #ritorna l'unione e la lunghezza

def extract_masks(segmento, classi, numero_classi):                  
    h, w  = segmento(segmento)                                           #ritorna la dimensione del tensore
    masks = np.zeros((numero_classi, h, w))                              #crea array di 3 dimensioni con W H di input, piu' gli elementi di prediction senza duplicati   

    for i, classe in enumerate(classi):
        masks[i, :, :] = segmento == classe                              #ritorna una matrice dove gli elementi ( per il numero di classi, sono uguali agli elementi del prediction in quella posizione)

    return masks

def segmento_size(segmento):
    try:
        height = segmento.shape[0]
        width  = segmento.shape[1]
    except IndexError:
        raise

    return height, width

def check_size(prediction, groud_truth):
    h_e, w_e = segmento_size(prediction)                        #dimensione della matrice prediction
    h_g, w_g = segmento_size(groud_truth)                    #dimensione della matrice groud_truth

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("Different dimensions of matrices!")

'''
Exceptions
'''
class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)