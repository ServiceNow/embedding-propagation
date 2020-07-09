import torch
from torch.nn import functional as F
from ..predict_methods import label_prop as lp
import numpy as np
from .. import utils as ut
from scipy import stats
from embedding_propagation import LabelPropagation

def ssl_get_next_best_indices(episode_dict, support_size_max=None):
    S = torch.from_numpy(episode_dict["support_so_far"]["samples"]).cuda()
    Q = torch.from_numpy(episode_dict["query"]["samples"]).cuda()
    U = torch.from_numpy(episode_dict["unlabeled"]["samples"]).cuda()
    S_labels = torch.from_numpy(episode_dict["support_so_far"]["labels"]).cuda()
    nclasses = int(S_labels.max() + 1)
    Q_labels = torch.zeros(episode_dict["query"]["samples"].shape[0], dtype=S_labels.dtype).cuda() + nclasses
    U_labels = torch.zeros(episode_dict["unlabeled"]["samples"].shape[0], dtype=S_labels.dtype).cuda() + nclasses
    A_labels = torch.cat([S_labels, Q_labels, U_labels], 0)
     
    SQU = torch.cat([S, Q, U], dim=0) # Information gain is measured in the whole system
    
    # train label_prop
    lp = LabelPropagation(balanced=True)
    logits = lp(SQU, A_labels, nclasses)

    U_logits = logits[-U.shape[0]:]
    # modify the labels of the unlabeled
    episode_dict["unlabeled"]["labels"] = U_logits.argmax(dim=1).cpu().numpy()

    # unlabeled_scores = U_logits.max(dim=1).cpu().numpy()
    
    if support_size_max is None:
        # choose all the unlabeled examples
        return np.arange(U.shape[0])
    else:
        # score each
        score_list = U_logits.max(dim=1)[0].cpu().numpy()
        return score_list.argsort()[-support_size_max:]

def predict(S, S_labels, UQ, U_shape):
    label_prop = lp.Labelprop()
    label_prop.fit(support_set=S, unlabeled_set=UQ)
    
    logits = label_prop.predict(support_labels=S_labels, 
                                unlabeled_pseudolabels=None,
                                balanced_flag=True)

    U_logits = logits[S.shape[0]:S.shape[0]+U_shape[0]]
    return U_logits.argmax(dim=1).cpu().numpy()