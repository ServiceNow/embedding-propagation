import torch
import numpy as np
import torch.nn.functional as F
import time

def _make_aligned_labels(inputs):
    batch, n_sample_pc, n_classes, z_dim = inputs.shape
    identity = torch.eye(n_classes, dtype=inputs.dtype, device=inputs.device)
    return identity[None, None, :, :].expand(batch, n_sample_pc, -1, -1).contiguous()

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