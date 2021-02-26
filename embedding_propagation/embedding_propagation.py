
import torch
import torch.nn.functional as F
import numpy as np


class EmbeddingPropagation(torch.nn.Module):
    def __init__(self, alpha=0.5, rbf_scale=1, norm_prop=False):
        super().__init__()
        self.alpha = alpha
        self.rbf_scale = rbf_scale
        self.norm_prop = norm_prop

    def forward(self, x, propagator=None):
        return embedding_propagation(x, self.alpha, self.rbf_scale, self.norm_prop, propagator=propagator)


class LabelPropagation(torch.nn.Module):
    def __init__(self, alpha=0.2, rbf_scale=1, norm_prop=True, apply_log=True, balanced=False):
        super().__init__()
        self.alpha = alpha
        self.rbf_scale = rbf_scale
        self.norm_prop = norm_prop
        self.apply_log = apply_log
        self.balanced = balanced

    def forward(self, x, labels, nclasses, propagator=None):
        """Applies label propagation given a set of embeddings and labels

        Arguments:
            x {Tensor} -- Input embeddings
            labels {Tensor} -- Input labels from 0 to nclasses + 1. The highest value corresponds to unlabeled samples.
            nclasses {int} -- Total number of classes

        Keyword Arguments:
            propagator {Tensor} -- A pre-computed propagator (default: {None})

        Returns:
            tuple(Tensor, Tensor) -- Logits and Propagator
        """
        return label_propagation(x, labels, nclasses, self.alpha, self.rbf_scale,
                                 self.norm_prop, self.apply_log, propagator=propagator,
                                 balanced=self.balanced)


def get_similarity_matrix(x, rbf_scale):
    b, c = x.size()
    sq_dist = ((x.view(b, 1, c) - x.view(1, b, c))**2).sum(-1) / np.sqrt(c)
    mask = sq_dist != 0
    sq_dist = sq_dist / sq_dist[mask].std()
    weights = torch.exp(-sq_dist * rbf_scale)
    mask = torch.eye(weights.size(1), dtype=torch.bool, device=weights.device)
    weights = weights * (~mask).float()
    return weights


def embedding_propagation(x, alpha, rbf_scale, norm_prop, propagator=None):
    if propagator is None:
        weights = get_similarity_matrix(x, rbf_scale)
        propagator = global_consistency(
            weights, alpha=alpha, norm_prop=norm_prop)
    return torch.mm(propagator, x)


def label_propagation(x, labels, nclasses, alpha, rbf_scale, norm_prop, apply_log, propagator=None, balanced=False, epsilon=1e-6):
    labels = F.one_hot(labels, nclasses + 1)
    labels = labels[:, :nclasses].float()  # the max label is unlabeled
    if balanced:
        labels = labels / labels.sum(0, keepdim=True)
    if propagator is None:
        weights = get_similarity_matrix(x, rbf_scale)
        propagator = global_consistency(
            weights, alpha=alpha, norm_prop=norm_prop)
    y_pred = torch.mm(propagator, labels)
    if apply_log:
        y_pred = torch.log(y_pred + epsilon)

    return y_pred


def global_consistency(weights, alpha=1, norm_prop=False):
    """Implements D. Zhou et al. "Learning with local and global consistency". (Same as in TPN paper but without bug)

    Args:
        weights: Tensor of shape (n, n). Expected to be exp( -d^2/s^2 ), where d is the euclidean distance and
            s the scale parameter.
        labels: Tensor of shape (n, n_classes)
        alpha: Scaler, acts as a smoothing factor
    Returns:
        Tensor of shape (n, n_classes) representing the logits of each classes
    """
    n = weights.shape[1]
    identity = torch.eye(n, dtype=weights.dtype, device=weights.device)
    isqrt_diag = 1. / torch.sqrt(1e-4 + torch.sum(weights, dim=-1))
    # checknan(laplacian=isqrt_diag)
    S = weights * isqrt_diag[None, :] * isqrt_diag[:, None]
    # checknan(normalizedlaplacian=S)
    propagator = identity - alpha * S
    propagator = torch.inverse(propagator[None, ...])[0]
    # checknan(propagator=propagator)
    if norm_prop:
        propagator = F.normalize(propagator, p=1, dim=-1)
    return propagator
