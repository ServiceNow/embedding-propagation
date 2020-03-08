# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F

# from models.wide_resnet_imagenet import WideResnetImagenet, Group
# from modules.dynamic_residual_groups import get_group
# from modules.ten import TEN
# from modules.distances import Distance, _propagate, standarized_label_prop
# from modules.output_heads import get_output_head
# from modules.activations import get_activation
# from modules.layers import MetricLinear
from . import label_prop as lp


def adaptive_predict(episode_dict, double_flag=False):
    # Get variables
    S = torch.from_numpy(episode_dict["support_so_far"]["samples"]).cuda()
    S_labels = torch.from_numpy(episode_dict["support_so_far"]["labels"]).cuda()
    Q = torch.from_numpy(episode_dict["query"]["samples"]).cuda()
    U = torch.from_numpy(episode_dict["unlabeled"]["samples"]).cuda()
    
    SUQ = torch.cat([S, U, Q])[None]
    Q_labels = -1*torch.ones(Q.shape[0], device=S_labels.device, dtype=S_labels.dtype).cuda()         
    U_labels = -1*torch.ones(U.shape[0], device=S_labels.device, dtype=S_labels.dtype).cuda()

    SUQ_labels = torch.cat([S_labels, U_labels, Q_labels])
    # Init Adaptive
    adaptive = AdaptiveTenInner(output_size=S.shape[1], 
                     nclasses=5, 
                     double_flag=double_flag).cuda()
   
    # Apply Adaptive
    # adaptive.train_step()
    UQ_logits = adaptive.forward(x=SUQ, 
                                 support_size=S.shape[0], 
                                 query_size=U.shape[0] + Q.shape[0], 
                                 labels=SUQ_labels)
    
    return UQ_logits[-Q.shape[0]:].argmax(dim=1)


class AdaptiveTenInner(torch.nn.Module):
    def __init__(self, output_size, nclasses, double_flag, mu_init=0, scale_init=0, precision_init=0, **exp_dict):
        super().__init__()
        self.nclasses = nclasses
        self.precision = torch.nn.Parameter(torch.randn(1, 1, output_size) * 0.1)
        self.classifier = torch.nn.Linear(output_size, nclasses)
        self.exp_dict = exp_dict
        self.optimizer = torch.optim.LBFGS([self.precision] + list(self.classifier.parameters()), tolerance_grad=1e-5, tolerance_change=1e-5, lr=0.1)
        if double_flag:
            self.label_prop = lp.LabelpropDouble()
        else:
            self.label_prop = lp.Labelprop()
        
    def train_step(self, _x, support_size, query_size, labels):
        x = _x.clone()
        self.optimizer.zero_grad()
        b, k, c = x.size()
        x = x * torch.sigmoid(1 + self.precision)

        zeros = torch.zeros(1, k, self.nclasses, device=x.device)

        logits, propagator = standarized_label_prop(x, zeros,
                                                    1, 1, 
                                                    apply_log=True, scale_bound="",
                                                    standarize="all", kernel="rbf")
        x = _propagate(x, propagator)
        # support_set = x.view(-1, self.nclasses, c)[:support_size, ...].view(-1, c)
        support_set = x.view(-1,  c)[:support_size, ...]
        support_labels = labels.view(-1, self.nclasses)[:support_size, ...].view(-1)
        logits = self.classifier(support_set)
        loss = F.cross_entropy(logits, support_labels) + 0.0001 * (self.precision ** 2).mean()
        loss.backward()
        return loss

    def forward(self, x, support_size, query_size, labels):
        self.train()
        self.optimizer.step(lambda: self.train_step(x, support_size, query_size, labels))
        # self.train_step(x, support_size, query_size, labels)
        with torch.no_grad():
            # mu, scale, precision = (0.5 + self.mu **2).detach(), (0.5 + self.scale ** 2).detach(), (1 + self.precision.detach())
            mu, scale, precision = 1, 1, self.precision.detach()
            # mu, scale, precision = 1, 1, 1
            x = x * torch.sigmoid(1 + precision)
            one_hot_labels = F.one_hot(labels.view(-1)).float()
            one_hot_labels = one_hot_labels.view(1,
                                                 support_size + query_size, 
                                                 self.nclasses)
            one_hot_labels[:, support_size:, ...] = 0

            logits, propagator = standarized_label_prop(x, one_hot_labels,
                                        scale, mu, apply_log=True,
                                        scale_bound="", standarize="all",
                                        kernel="rbf")
            x = _propagate(x, propagator)
            logits, propagator = standarized_label_prop(x, one_hot_labels,
                                        scale, mu, apply_log=True,
                                        scale_bound="", standarize="all",
                                        kernel="rbf")
            logits = logits.view((support_size + query_size), self.nclasses)
            return logits[support_size:].view(-1, self.nclasses)


def standarized_label_prop(embeddings,
                           labels,
                           gaussian_scale=1, alpha=1,
                           weights=None,
                           apply_log=False, scale_bound="", standarize="", kernel="", square_root=False,
                           epsilon=1e-6):
    if scale_bound == "softplus":
        gaussian_scale = 0.01 + F.softplus(gaussian_scale)
        alpha = 0.1 + F.softplus(alpha)
    elif scale_bound == "square":
        gaussian_scale = 1e-4 + gaussian_scale ** 2
        alpha = 0.1 + alpha ** 2
    elif scale_bound == "convex_relu":
        #gaussian_scale = gaussian_scale ** 2
        alpha = F.relu(alpha) + 0.1
    elif scale_bound == "convex_square":
        # gaussian_scale = gaussian_scale ** 2
        alpha = 0.1 + alpha ** 2
    elif scale_bound == "relu":
        gaussian_scale = F.relu(gaussian_scale) + 0.01
        alpha = F.relu(alpha) + 0.1
    elif scale_bound == "constant":
        gaussian_scale = 1
        alpha = 1
    elif scale_bound == "alpha_square":
        alpha = 0.1 + F.relu(alpha)

    # Compute the pairwise distance between the examples of the sample and query sets
    # XXX: labels are set to a constant for the query set
    sq_dist = generalized_pw_sq_dist(embeddings, "euclidean")
    if square_root:
        sq_dist = (sq_dist + epsilon).sqrt()
    if standarize == "all":
        mask = sq_dist != 0
     #   sq_dist = sq_dist - sq_dist[mask].mean()
        sq_dist = sq_dist / sq_dist[mask].std()
    elif standarize == "median":
        mask = sq_dist != 0
        gaussian_scale = torch.sqrt(
            0.5 * torch.median(sq_dist[mask]) / torch.log(torch.ones(1, device=sq_dist.device) + sq_dist.size(1)))
    elif standarize == "frobenius":
        mask = sq_dist != 0
        sq_dist = sq_dist / (sq_dist[mask] ** 2).sum().sqrt()
    elif standarize == "percentile":
        mask = sq_dist != 2
        sorted, indices = torch.sort(sq_dist.data[mask])
        total = sorted.size(0)
        gaussian_scale = sorted[int(total * 0.1)].detach()

    if kernel == "rbf":
        weights = torch.exp(-sq_dist * gaussian_scale)
    elif kernel == "convex_rbf":
        scales = torch.linspace(0.1, 10, gaussian_scale.size(0), device=sq_dist.device, dtype=sq_dist.dtype)
        weights = torch.exp(-sq_dist.unsqueeze(1) * scales.view(1, -1, 1, 1))
        weights = (weights * F.softmax(gaussian_scale.view(1, -1, 1, 1), dim=1)).sum(1)
        # checknan(timessoftmax=weights)
    elif kernel == "euclidean":
        # Compute similarity between the examples -- inversely proportional to distance
        weights = 1 / (gaussian_scale + sq_dist)
    elif kernel == "softmax":
        weights = F.softmax(-sq_dist / gaussian_scale, -1)

    mask = torch.eye(weights.size(1), dtype=torch.bool, device=weights.device)[None, :, :]
    weights = weights * (~mask).float()

    logits, propagator = global_consistency(weights, labels, alpha=alpha)

    if apply_log:
        logits = torch.log(logits + epsilon)

    return logits, propagator

def generalized_pw_sq_dist(data, d_type="euclidean"):
    batch, samples, z_dim = data.size()
    if d_type == "euclidean":
        return torch.sum((data[:, :, None, :] - data[:, None, :, :]) ** 2, dim=3) / np.sqrt(z_dim)
    elif d_type == "l1":
        return torch.mean(torch.abs(data[:, :, None, :] - data[:, None, :, :]), dim=3)
    elif d_type == "stable_euclidean":
        return torch.sqrt(1e-6 + torch.mean((data[:, :, None, :] - data[:, None, :, :]) ** 2, dim=3) / np.sqrt(z_dim))
    elif d_type == "cosine":
        data = F.normalize(data, dim=2)
        return torch.bmm(data, data.transpose(2, 1))
    else:
        raise ValueError("Distance type not recognized")

def global_consistency(weights, labels, alpha=0.1):
    """Implements D. Zhou et al. "Learning with local and global consistency". (Same as in TPN paper but without bug)
    Args:
        weights: Tensor of shape (batch, n, n). Expected to be exp( -d^2/s^2 ), where d is the euclidean distance and
            s the scale parameter.
        labels: Tensor of shape (batch, n, n_classes)
        alpha: Scaler, acts as a smoothing factor
    Returns:
        Tensor of shape (batch, n, n_classes) representing the logits of each classes
    """
    n = weights.shape[1]
    _alpha = 1 / (1 + alpha)
    _beta = alpha / (1 + alpha)
    identity = torch.eye(n, dtype=weights.dtype, device=weights.device)[None, :, :]
    #weights = weights * (1. - identity)  # zero out diagonal
    isqrt_diag = 1. / torch.sqrt(1e-4 + torch.sum(weights, dim=2))
    # checknan(laplacian=isqrt_diag)
    S = weights * isqrt_diag[:, None, :] * isqrt_diag[:, :, None]
    # checknan(normalizedlaplacian=S)
    propagator = identity - _alpha * S
    propagator = torch.inverse(propagator) * _beta
    # checknan(propagator=propagator)

    return _propagate(labels, propagator, scaling=1), propagator


def _propagate(labels, propagator, scaling=1.):
    return torch.matmul(propagator, labels) * scaling