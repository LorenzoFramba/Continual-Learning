import numpy as np
import torch


    #numero di pixel correttamente classificati / numero di erroreamente non pixel classificati ( falsi negativi)
def pixel_acc( mask, predicted, total_train, correct_train):
    total_train += mask.nelement()          #conta gli elementi all'interno del tensore
    train_accuracy = 100 * correct_train / total_train
    return train_accuracy, total_train, correct_train

def overall_pixel_acc( matrix):
    correct = torch.diag(matrix).sum()                      # trasforma la matrice in tensore dove gli input sono la diagonale, e ci fa la somma
    total = matrix.sum()                                    # fa la somma degli elementi della matrice
    overall_acc = correct * 100 / (total)                  #+ 1e-10  rapporto tra i corretti e i totali, 
    return overall_acc

def nanmean(x):
    return torch.mean(x[x == x])

def nanmax(x):
    return torch.max(x[x == x])

def mean_IU_2(matrix):
    A_inter_B = torch.diag(matrix)
    A = matrix.sum(dim=1)                                   #somma tutti gli elementi, sempre divisi in classi
    B = matrix.sum(dim=0)                                   #somma tutti i primi elementi
    jaccard = A_inter_B / (A + B - A_inter_B )##+ 1e-10
    avg_jacc = nanmean(jaccard)
    return avg_jacc

#  crea una matrice, che ha come lato la somma tra target e prediction
def _fast_conf_matrix(target, prediction, num_classes):
    mask = (target >= 0) & (target < num_classes)
    conf_matrix = torch.bincount(
        num_classes * target[mask] + prediction[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    return conf_matrix

#pixel accuracy per ogni classe 
def per_class_pixel_acc(conf_matrix):
    correct_per_class = torch.diag(conf_matrix)             #ritorna gli elementi in diagonale della matrice
    total_per_class = conf_matrix.sum(dim=1)                # fa la somma degli elementi, mantenendogli in un array
    per_class_acc = 100* correct_per_class / (total_per_class )##+ 1e-10
    avg_per_class_acc = nanmean(per_class_acc)
    return avg_per_class_acc

def max_per_class_pixel_acc(conf_matrix):
    correct_per_class = torch.diag(conf_matrix)             #ritorna gli elementi in diagonale della matrice
    total_per_class = conf_matrix.sum(dim=1)                # fa la somma degli elementi, mantenendogli in un array
    per_class_acc = 100* correct_per_class / (total_per_class )##+ 1e-10
    max_per_class_acc = nanmax(per_class_acc)
    return max_per_class_acc

def eval_metrics(target, prediction, num_classes):
    matrix = torch.zeros((num_classes, num_classes))
    for t, p in zip(target, prediction):
        matrix += _fast_conf_matrix(t.flatten(), p.flatten(), num_classes)  #confusion matrix
    overall_acc = overall_pixel_acc(matrix)
    avg_per_class_acc = per_class_pixel_acc(matrix)
    max_per_class_acc = max_per_class_pixel_acc(matrix)
    mean_IU = mean_IU_2(matrix)
    return overall_acc, avg_per_class_acc,mean_IU,max_per_class_acc


#pixel accuracy * 1 / ( sommatoria ( falsi negativi - pixel correttamente classificati))  * 1 / numero di classi   ``#        """ Calculate mean Intersection over Union """
def mean_IU_(target, prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def pixel_accuracy(eval_segm, gt_segm):
  
  #  sum_i(n_ii) / sum_i(t_i) #
    

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i  = 0

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i  += np.sum(curr_gt_mask)
 
    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_

def mean_accuracy(eval_segm, gt_segm):

    #    (1/n_cl) sum_i(n_ii/t_i)    #
    

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
 
        if (t_i != 0):
            accuracy[i] = n_ii / t_i

    mean_accuracy_ = np.mean(accuracy)
    return mean_accuracy_

def mean_IU(eval_segm, gt_segm):
    
    # (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))# 
    

    check_size(eval_segm, gt_segm)

    cl, n_cl   = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
 
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)
 
    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_

def frequency_weighted_IU(eval_segm, gt_segm):
    
    # sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))# 

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    frequency_weighted_IU_ = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
 
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)
 
    sum_k_t_k = get_pixel_area(eval_segm)
    
    frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
    return frequency_weighted_IU_


#Auxiliary functions used during evaluation. #

def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]

def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask   = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask

def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl

def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _   = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl

def extract_masks(segm, cl, n_cl):
    h, w  = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks

def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise

    return height, width

def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")

'''
Exceptions
'''
class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)