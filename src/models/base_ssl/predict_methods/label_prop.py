
import matplotlib.pyplot as plt
# import seaborn as sns
import torch
import torch.nn.functional as F
import numpy as np
import warnings
from functools import partial
# from tools.meters import BasicMeter
# from train import TensorLogger
import sys
from embedding_propagation import LabelPropagation

def label_prop_predict(episode_dict, double_flag=False):
    S = torch.from_numpy(episode_dict["support_so_far"]["samples"]).cuda()
    S_labels = torch.from_numpy(episode_dict["support_so_far"]["labels"]).cuda()
    nclasses = int(S_labels.max() + 1)
    Q_labels = torch.zeros(episode_dict["query"]["samples"].shape[0], dtype=S_labels.dtype).cuda() + nclasses
    U_labels = torch.zeros(episode_dict["unlabeled"]["samples"].shape[0], dtype=S_labels.dtype).cuda() + nclasses
    A_labels = torch.cat([S_labels, Q_labels, U_labels], 0)
    Q = torch.from_numpy(episode_dict["query"]["samples"]).cuda()

    U = torch.from_numpy(episode_dict["unlabeled"]["samples"]).cuda()
        
    lp = LabelPropagation(balanced=True)

    SUQ = torch.cat([S, U, Q], dim=0)
    logits = lp(SUQ, A_labels, nclasses)
    logits_query = logits[-Q.shape[0]:]

    return logits_query.argmax(dim=1).cpu().numpy()

class Labelprop():
    def __init__(self, mu=1, scale=1, n_classes=5):
        self.scale = scale
        self.mu = mu
        self.n_classes = n_classes

    def fit(self, support_set, unlabeled_set):
        # make assertions and get info
        assert(len(support_set.size()) == 2)
        assert(len(unlabeled_set.size()) == 2)
        data = torch.cat([support_set, unlabeled_set], 0)
        self.n_samples_all = data.shape[0]
        b, c = data.size()

        # Compute weight matrix
        data_left = data.view(1, b, c)
        data_right = data.view(b, 1, c)
        sq_dist = ((data_left.expand(b, b, c) - data_right.expand(b, b, c)) ** 2).sum(-1) / np.sqrt(c)
        mask = ~(torch.eye(b, dtype=sq_dist.dtype, device=sq_dist.device).bool())
        sq_dist = sq_dist / sq_dist[mask].std()

        # Apply kernel
        weights = torch.exp(-sq_dist * self.scale)
        weights = weights * mask.float()
        self.adjacency = weights

        # Compute propagator
        alpha_ = 1 / (1 + self.mu)
        beta_ = self.mu / (1 + self.mu)
        n = weights.shape[1]
        isqrt_diag = 1. / torch.sqrt(1e-4 + torch.sum(weights, dim=1))
        S = weights * isqrt_diag[None, :] * isqrt_diag[:, None]
        identity = torch.eye(n, dtype=weights.dtype, device=weights.device)
        propagator = identity - alpha_ * S
        self.propagator = torch.inverse(propagator) * beta_

        assert(list(self.propagator.size()) == [b, b])
        
        return self.propagator
        

    def predict(self, support_labels, unlabeled_pseudolabels=None, balanced_flag=False):
        unlabeled_one_hot = torch.zeros(self.n_samples_all-support_labels.shape[0], 
                                         self.n_classes, 
                                         dtype=support_labels.dtype,
                                         device=support_labels.device)
                                         
        if unlabeled_pseudolabels is not None:
            for cls, ind in zip(unlabeled_pseudolabels["classes"], 
                                unlabeled_pseudolabels["indices"]):
                unlabeled_one_hot[ind, cls] = 1

        one_hot = torch.cat([F.one_hot(support_labels, num_classes=self.n_classes), 
                             unlabeled_one_hot]).float()

        assert(one_hot.ndim==2)
        if balanced_flag:
            one_hot = one_hot / one_hot.sum(0, keepdim=True)

        preds = torch.matmul(self.propagator, one_hot) * 1.
        logits = torch.log(preds)

        assert(~torch.isnan(torch.log(preds)).any())
        return logits

class LabelpropDouble(Labelprop):
    def __init__(self, mu=1, scale=1, n_classes=5):
        super().__init__(mu=mu, scale=scale, n_classes=n_classes)
        
    def fit(self,  support_set, unlabeled_set):
        S = support_set
        U = unlabeled_set

        propagator = super().fit(S, U)
        SU = torch.cat([support_set, unlabeled_set], 0)
        SU_p = torch.matmul(propagator, SU)

        # split data again into support and unlabeled
        S_new = SU_p[:S.shape[0]]
        U_new = SU_p[S.shape[0]:]
        propagator_new = super().fit(S_new, U_new)

        return propagator_new