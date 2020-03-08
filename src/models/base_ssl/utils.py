
# %% Import libraries & loading data & Helper Sampling Class &  Pytorch helpers and imports
import torch
import sys
import copy
from scipy.stats import pearsonr
import h5py
import os
import numpy as np
import pylab
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
# from ipywidgets import interact, interactive, fixed, interact_manual
# import ipywidgets as widgets
import torch
import torch.nn.functional as F
from functools import partial
import json
from skimage.io import imsave
import tqdm
import pprint
import torch
import sys
from .distances import prototype_distance # here we import the labelpropagation algorithm inside the "Distance" class
import pandas
import json  

# loading data
# from trainers.few_shot_parallel_alpha_scale import inner_loop_lbfgs2, inner_loop_lbfgs_bootstrap
from torch.utils.data import DataLoader

    

def get_unlabeled_set(method, unlabeled_set,unlabel_labels, unlabeled_size,
                      support_set_reshaped):
    support_labels = get_support_labels(support_set_reshaped).ravel().astype(int)
    support_size = support_set_reshaped.shape[1]
    n_classes = support_set_reshaped.shape[0]
    if method == "cheating":
            unlabeled_set_new = unlabeled_set
            unlabel_labels_new = unlabel_labels
            unlabeled_size_new = unlabeled_size


    elif method == "prototypical":
        support_labels_reshaped = support_labels.reshape((n_classes, support_size))
        unlabeled_set_torch = torch.FloatTensor(unlabeled_set).cuda()
        unlabeled_set_view = unlabeled_set_torch.view(1, n_classes, unlabeled_size, a512)

        if True:
            unlabeled_set_view, tmp_unlabel_labels = predict_sort(support_set_reshaped, 
                                        unlabeled_set_view, 
                                        n_classes=n_classes)
        else:
            tmp_unlabel_labels = predict(support_set_reshaped, 
                                        unlabeled_set_view, 
                                        n_classes=n_classes)

        unlabeled_set_list = []
        # label
        unlabeled_size_new = np.inf
        for c in range(n_classes):
            set = unlabeled_set_torch[tmp_unlabel_labels == c]
            unlabeled_set_list += [set]
            unlabeled_size_new = min(len(set), unlabeled_size_new)

        # cut to shortest
        unlabel_labels_new = []
        for c in range(n_classes):
            unlabeled_set_list[c] = unlabeled_set_list[c][:unlabeled_size_new]
            unlabel_labels_new += [np.ones(unlabeled_size_new) * c]

        unlabeled_set_new = torch.cat(unlabeled_set_list).cpu().numpy().astype(unlabeled_set.dtype)
        unlabel_labels_new = np.vstack(unlabel_labels_new).ravel().astype("int64")

    return unlabeled_set_new, unlabel_labels_new, unlabeled_size_new 




def xlogy(x, y=None):
    z = torch.zeros(())
    if y is None:
        y = x
    assert y.min() >= 0
    return x * torch.where(x == 0., z, torch.log(y))


def get_support_labels(support_set_features):
    support_labels=[] 
    for c in range(5):
        support_labels += [np.ones(support_set_features.shape[1])*c]

    support_labels = np.vstack(support_labels)
    return support_labels    


def get_entropy_support_set(monitor, support_size):
    support_set_ind = [[],[],[],[],[]]
    for s in range(1, support_size+1):
        # Get best next support
        entropy_best = 0
        for i in range(monitor.unlabeled_size):
            support_set_tmp = copy.deepcopy(support_set_ind)
            if i in support_set_tmp[0]:
                continue
            for c in range(monitor.n_classes):
                s_ind = monitor.unlabeled_size*c
                ind = s_ind + i
                
                support_set_tmp[c] += [ind]
            
            entropy_tmp = monitor.compute_entropy(support_set_tmp)      
            if entropy_tmp >= entropy_best:
                support_set_best = support_set_tmp
                entropy_best = entropy_tmp

        support_set_ind = support_set_best

    for c in range(monitor.n_classes):
        ind_c = np.arange(monitor.unlabeled_size*c, monitor.unlabeled_size*(c+1))
        # 1. Within class ind sanity check
        assert False not in [sc in ind_c for sc in support_set_ind[c]] 
        # 2. Uniqueness sanity check
        assert np.unique(support_set_ind[c]).size == np.array(support_set_ind[c]).size

    support_set_list = [monitor.unlabeled_set[i_list] for i_list in support_set_ind]
    support_set = np.vstack(support_set_list) 
    return support_set

def get_kmeans_support_set(monitor, support_size):
    # unlabeled_set = unlabeled_set
    # unlabel_labels = unlabel_labels

    # greedy
    support_set_list = [[],[],[],[],[]]
    for c in range(monitor.n_classes):
        s_ind = monitor.unlabeled_size*c

        X = monitor.unlabeled_set[s_ind:s_ind+monitor.unlabeled_size]
        k_means = KMeans(n_clusters=support_size, random_state=0).fit(X)
        support_set_list[c] = k_means.cluster_centers_
    support_set = np.vstack(support_set_list)
    return support_set


        
def get_greedy_support_set(monitor, support_size):
    # unlabeled_set = unlabeled_set
    # unlabel_labels = unlabel_labels

    # greedy
    support_set_ind = [[],[],[],[],[]]
    for s in range(1, support_size+1):
        # Get best next support
        acc_best = 0.
        for i in range(monitor.unlabeled_size):
            support_set_tmp = copy.deepcopy(support_set_ind)
            if i in support_set_tmp[0]:
                continue
            for c in range(monitor.n_classes):
                s_ind = monitor.unlabeled_size*c
                ind = s_ind + i
                
                support_set_tmp[c] += [ind]
            
            acc_tmp = monitor.compute_acc(support_set_tmp)      
            if acc_tmp >= acc_best:
                support_set_best = support_set_tmp
                acc_best = acc_tmp

        support_set_ind = support_set_best

    for c in range(monitor.n_classes):
        ind_c = np.arange(monitor.unlabeled_size*c, monitor.unlabeled_size*(c+1))
        # 1. Within class ind sanity check
        assert False not in [sc in ind_c for sc in support_set_ind[c]] 
        # 2. Uniqueness sanity check
        assert np.unique(support_set_ind[c]).size == np.array(support_set_ind[c]).size

    support_set_list = [monitor.unlabeled_set[i_list] for i_list in support_set_ind]
    support_set = np.vstack(support_set_list) 
    return support_set




def get_random_support_set(monitor, support_size):
    support_set_list = []
    for c in range(monitor.n_classes):
        ind = np.arange(monitor.unlabeled_size*c, monitor.unlabeled_size*(c+1))
        ind_c = np.random.choice(ind, support_size, replace=False)
        support_set_list += [monitor.unlabeled_set[ind_c]]

    support_set = np.vstack(support_set_list)
    return support_set

@torch.no_grad()
def calc_accuracy(split, iters, distance_fn, support_size, 
                query_size, unlabeled_size, method=None, model=None, n_classes=5):
    sampler = Sampler(split)
    accuracies = []

    if method == "pairs":
        check_pairs(split, iters, distance_fn, support_size, 
                    query_size, unlabeled_size, n_classes=5)
   



# Helper Sampling Class
def make_labels(size, classes):
    """
    Helper function. Generates the labels of a set: e.g 0000 1111 2222 for size=4, classes=3
    """
    return (np.arange(classes).reshape((classes, 1)) + np.zeros((classes, size), dtype=np.int32)).flatten()

# Pytorch helpers and imports  
def to_pytorch(datum, n, k, c):
    if k > 0:
        return torch.from_numpy(datum).view(-1, n, k, c)
    else:
        return None
