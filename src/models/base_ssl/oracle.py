from . import utils as ut
import numpy as np 
import torch
import sys
import copy
from scipy.stats import pearsonr
import h5py
import os
import numpy as np
import pylab

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
sys.path.insert(0, os.path.abspath('..'))
from .distances import prototype_distance # here we import the labelpropagation algorithm inside the "Distance" class
import pandas
import json  

# loading data
# from trainers.few_shot_parallel_alpha_scale import inner_loop_lbfgs2, inner_loop_lbfgs_bootstrap
from torch.utils.data import DataLoader

class Sampler(object):
    """
    Samples few shot tasks from precomputed embeddings
    """
    def __init__(self, embeddings_fname, n_classes, distract_flag):
        self.h5fp = h5py.File(embeddings_fname, 'r')
        self.labels = self.h5fp["test_targets"][...]
        indices = np.arange(self.labels.shape[0])
        self.label_indices = {i: indices[self.labels == i] for i in set(self.labels)}
        self.nclasses = len(self.label_indices.keys())
        self.n_classes = n_classes
        self.distract_flag = distract_flag

    def sample_episode_indices(self, support_size,
     query_size, unlabeled_size, ways):
        """
        Returns the indices of the images of a random episode with predefined support, query and unlabeled sizes.
        the number of images is expressed in "ways"
        """
        label_indices = {k: np.random.permutation(v) for k,v in self.label_indices.items()}
        #label_indices = self.label_indices
        
        if self.distract_flag:
            classes = np.random.permutation(self.nclasses)
            distract_classes = classes[ways:(ways+ways)]
            classes = classes[:ways]
        else:
            classes = np.random.permutation(self.nclasses)[:ways]

        support_indices = []
        query_indices = []
        unlabel_indices = []
        
        for cls in classes:
            start = 0
            end = support_size
            support_indices.append(label_indices[cls][start:end])
            start = end
            end += query_size
            query_indices.append(label_indices[cls][start:end])
            start = end
            end += unlabeled_size
            assert(end < len(label_indices[cls]))
            unlabel_indices.append(label_indices[cls][start:end])

        if self.distract_flag:
            for cls in distract_classes:
                unlabel_indices.append(label_indices[cls][:unlabeled_size])

        return np.vstack(support_indices), np.vstack(query_indices), np.vstack(unlabel_indices)
    
    def _sample_field(self, field, *indices):
        features = self.h5fp["test_{}".format(field)]
        ret = []
        for _indices in indices:
            _indices = _indices.ravel()
            argind = np.argsort(_indices)
            if len(argind) == 0:
                ret.append(None)
            else:
                ind = _indices[argind]
                dset = features[ind.tolist()]
                dset[argind] = dset.copy()
                ret.append(dset)

        return tuple(ret)
    
    def sample_features(self, support_indices, query_indices, unlabel_indices):
        return self._sample_field("features", support_indices, query_indices, unlabel_indices)
    
    def sample_labels(self, support_indices, query_indices, unlabel_indices):
        return self._sample_field("targets", support_indices, query_indices, unlabel_indices)
    
    def sample_episode(self, support_size, query_size, unlabeled_size, apply_ten_flag=False):
        """
        Randomly samples an episode (features and labels) given the size of each set and the number of classes
        
        Returns: tuple(numpy array). Sets are of the size (set_size * nclasses, a512). a512 is the number
                    of channels of the embeddings
        """
        ways = self.n_classes
        support_indices, query_indices, unlabel_indices = self.sample_episode_indices(support_size, query_size, unlabeled_size, ways)
        support_set, query_set, unlabel_set = self.sample_features(support_indices, 
                                                                   query_indices, 
                                                                   unlabel_indices)
        support_labels = ut.make_labels(support_size, ways)
        query_labels =  ut.make_labels(query_size, ways)
        unlabel_labels =  ut.make_labels(unlabeled_size, ways)
        
        episode_dict = episode2dict(support_set, query_set, unlabel_set, support_labels, query_labels, unlabel_labels)
        
        if apply_ten_flag:
            episode_dict = ut.apply_ten_on_episode(episode_dict)

        return episode_dict

def episode2dict(support_set, query_set, unlabel_set, support_labels, query_labels, unlabel_labels):
    n_classes = support_set.shape[0]
    
    support_dict = {"samples": support_set, "labels":support_labels}
    query_dict = {"samples": query_set, "labels":query_labels}
    unlabeled_dict = {"samples": unlabel_set, "labels":unlabel_labels}

    return {"support":support_dict, 
            "query":query_dict,
            "unlabeled":unlabeled_dict}



def compute_acc(pred_labels, true_labels):
    acc = (true_labels.flatten() == pred_labels.ravel()).astype(float).mean()

    return acc

    

    

