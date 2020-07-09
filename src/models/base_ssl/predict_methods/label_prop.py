
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

def label_prop_predict(episode_dict):
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