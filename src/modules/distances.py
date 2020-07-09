import torch
import numpy as np
import torch.nn.functional as F
def _make_aligned_labels(inputs):
    batch, n_sample_pc, n_classes, z_dim = inputs.shape
    identity = torch.eye(n_classes, dtype=inputs.dtype, device=inputs.device)
    return identity[None, None, :, :].expand(batch, n_sample_pc, -1, -1).contiguous()
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
def standarized_label_prop(embeddings,
                           labels,
                           gaussian_scale=1, alpha=0.5,
                           weights=None,
                           apply_log=False, 
                           scale_bound="", 
                           standarize="all", 
                           kernel="rbf", 
                           square_root=False,
                           norm_prop=0,
                           epsilon=1e-6):
    propagator_scale = gaussian_scale
    gaussian_scale = 1
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
    logits, propagator = global_consistency(weights, labels, alpha=alpha, norm_prop=norm_prop, scale=propagator_scale)
    if apply_log:
        logits = torch.log(logits + epsilon)
    return logits, propagator
def global_consistency(weights, labels, alpha=1, norm_prop=0, scale=1):
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
    identity = torch.eye(n, dtype=weights.dtype, device=weights.device)[None, :, :]
    #weights = weights * (1. - identity)  # zero out diagonal
    isqrt_diag = 1. / torch.sqrt(1e-4 + torch.sum(weights, dim=2))
    # checknan(laplacian=isqrt_diag)
    S = weights * isqrt_diag[:, None, :] * isqrt_diag[:, :, None]
    # checknan(normalizedlaplacian=S)
    propagator = identity - alpha * S
    propagator = torch.inverse(propagator)
    # checknan(propagator=propagator)
    if norm_prop > 0:
        propagator = F.normalize(propagator, p=norm_prop, dim=-1)
    elif norm_prop < 0:
        propagator = F.softmax(propagator, dim=-1)
    propagator = propagator * scale
    return _propagate(labels, propagator, scaling=1), propagator
def _propagate(labels, propagator, scaling=1.):
    return torch.matmul(propagator, labels) * scaling
def prototype_distance(support_set, query_set, labels, unlabeled_set=None):
  """Computes distance from each element of the query set to prototypes in the sample set.
  Args:
      sample_set: Tensor of shape (batch, n_classes, n_sample_per_classes, z_dim) containing the representation z of
          each images.
      query_set: Tensor of shape (batch, n_classes, n_query_per_classes, z_dim) containing the representation z of
          each images.
      unlabeled_set: Tensor of shape (batch, n_classes, n_unlabeled_per_classes, z_dim) containing the representation
          z of each images.
  Returns:
      Tensor of shape (batch, n_total_query, n_classes) containing the similarity between each pair of query,
      prototypes, for each task.
  """
  n_queries, channels = query_set.size()
  n_support, channels = support_set.size()
  support_set = support_set.view(n_support, 1, channels)
  way = int(labels.data.max()) + 1
  one_hot_labels = torch.zeros(n_support, way, 1, dtype=support_set.dtype, device=support_set.device)
  one_hot_labels.scatter_(1, labels.view(n_support, 1, 1), 1)
  total_per_class = one_hot_labels.sum(0, keepdim=True)
  prototypes = (support_set * one_hot_labels).sum(0) / total_per_class
  prototypes = prototypes.view(1, way, channels)
  query_set = query_set.view(n_queries, 1, channels)
  d = query_set - prototypes
  return -torch.sum(d ** 2, 2) / np.sqrt(channels)