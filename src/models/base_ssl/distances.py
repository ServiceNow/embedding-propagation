
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

def checknan(**kwargs):
    for item, value in kwargs.items():
        v = value.data.sum()
        if torch.isnan(v) or torch.isinf(torch.abs(v)):
            return "%s is NaN or inf" %item

def _make_aligned_labels(inputs):
    batch, n_classes, n_sample_pc, z_dim = inputs.shape
    identity = torch.eye(n_classes, dtype=inputs.dtype, device=inputs.device)
    return identity[None, :, None, :].expand(batch, -1, n_sample_pc, -1).contiguous()


def matching_nets(support_set, query_set, *sets, distance_type="euclidean", **kwargs):
    """ Computes the logits using the method described in [1]

    [1] Vinyals, Oriol, et al. "Matching networks for one shot learning." NeurIPS. 2016.

    Args:
        sample_set: Tensor of shape (batch, n_classes, n_sample_per_classes, z_dim) containing the representation z of
            each images.
        query_set: Tensor of shape (batch, n_classes, n_query_per_classes, z_dim) containing the representation z of
            each images.
        unlabeled_set: Tensor of shape (batch, n_classes, n_unlabeled_per_classes, z_dim) containing the representation
            z of each images.
        euclidean: Whether to use the euclidean distance or the cosine distance

    Returns:
        Class logits (warning, this function returns log probabilities, so NLL loss is recommended)
    """
    euclidean = distance_type == "euclidean"
    _support_set, _support_labels = support_set
    _query_set, _query_labels = query_set
    b, n, sk, z = _support_set.size()
    b, n, qk, z = _query_set.size()
    if isinstance(_support_labels, bool):
        labels = _make_aligned_labels(_support_set)
    else:
        labels = _support_labels
    if euclidean:
        _support_set = _support_set.view(b, 1, n * sk, z)
        _query_set = _query_set.view(b, n * qk, 1, z)
        att = - ((_support_set - _query_set) ** 2).sum(3) / np.sqrt(z)
    else:
        _support_set = F.normalize(_support_set, dim=3)
        _query_set = F.normalize(_query_set, dim=3)
        _support_set = _support_set.view(b, n * sk, z).transpose(2, 1)
        _query_set = _query_set.view(b, n * qk, z)
        att = torch.matmul(_query_set, _support_set)
    att = F.softmax(att, dim=2).view(b, n * qk, 1, n * sk)
    labels = labels.view(b, 1, n * sk, n)
    return torch.log(torch.matmul(att, labels).view(b * n * qk, n))


def prototype_distance(support_set, query_set, *args, **kwargs):
    """Computes distance from each element of the query set to prototypes in the sample set.

    Args:
        sample_set: tuple of (Tensor, is_labeled=True). The Tensor has shape
            (batch, n_classes, n_sample_per_classes, z_dim) containing the representation z of each images.
        query_set: tuple of (Tensor, is_labeled=False). The tensor has shape,
            (batch, n_classes, n_query_per_classes, z_dim) containing the representation z of each images.

    Returns:
        Tensor of shape (batch, n_total_query, n_classes) containing the similarity between each pair of query,
        prototypes, for each task.
    """
    _support_set, _support_labels = support_set
    _query_set, _query_labels = query_set
    b, n, query_size, c = _query_set.size()
    _support_set = _support_set.mean(2).view(b, 1, n, c)
    _query_set = _query_set.view(b, n * query_size, 1, c)
    d = _query_set - _support_set
    return -torch.sum(d ** 2, 3) / np.sqrt(c)


def gauss_distance(sample_set, query_set, unlabeled_set=None):
    """ (experimental) function to try different approaches to model prototypes as gaussians
    Args:
        sample_set: features extracted from the sample set
        query_set: features extracted from the query set
        query_set: features extracted from the unlabeled set

    """
    b, n, k, c = sample_set.size()
    sample_set_std = sample_set.std(2).view(b, 1, n, c)
    sample_set_mean = sample_set.mean(2).view(b, 1, n, c)
    query_set = query_set.view(b, n * k, 1, c)
    d = (query_set - sample_set_mean) / sample_set_std
    return -torch.sum(d ** 2, 3) / np.sqrt(c)


def _make_aligned_labels(inputs):
    """Uses the shape of inputs to infer batch_size, n_classes, and n_sample_per_class. From this, we build the one-hot
    encoding label tensor aligned with inputs. This is used to keep the lable information when tensors are flatenned
    across the n_class and n_sample_per_class.

    Args:
        inputs: tensor of shape (batch, n_classes, n_sample_per_class, z_dim) containing encoded examples for a task.
    Returns:
        tensor of shape (batch, n_classes, n_sample_pc, n_classes) containing the one-hot encoding label of each example
    """
    batch, n_classes, n_sample_pc, z_dim = inputs.shape
    identity = torch.eye(n_classes, dtype=inputs.dtype, device=inputs.device)
    return identity[None, :, None, :].expand(batch, -1, n_sample_pc, -1).contiguous()


def generalized_pw_sq_dist(data, d_type="euclidean"):
    batch, n_classes, n_samples, z_dim = data.size()
    data = data.view(batch, -1, z_dim)
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


def pw_sq_dist(sample_set, query_set, unlabeled_set=None, label_offset=0):
    """Computes distance from each element of the query set to prototypes in the sample set.

    Args:
        sample_set: Tensor of shape (batch, n_classes, n_sample_per_class, z_dim) containing the representation z of
            each images.
        query_set: Tensor of shape (batch, n_classes, n_query_per_class, z_dim) containing the representation z of
            each images.

    Returns:
        dist: Tensor of shape (batch, n_total, n_total) containing the squared distance between each pair.
        labels: Tensor of shape (batch, n_total, n_classes) Containing the one hot vector for the sample set and zeros
            for the query set.
    """
    batch, n_classes, _, z_dim = sample_set.shape
    sample_labels = _make_aligned_labels(sample_set)
    query_labels = _make_aligned_labels(query_set) * 0. + label_offset  # XXX: Set labels to a constant
    # TODO: it's a bit sketchy that this function is used to extract the labels. Should be done externally.
    labels = torch.cat((sample_labels, query_labels), dim=2).view(batch, -1, n_classes)
    samples = torch.cat((sample_set, query_set), dim=2).view(batch, -1, z_dim)
    return torch.sum((samples[:, :, None, :] - samples[:, None, :, :]) ** 2, dim=3) / np.sqrt(z_dim), labels


def _ravel_index(index, shape):
    shape0 = np.prod(shape[:-1])
    shape1 = shape[-1]
    new_shape = list(shape[:-1]) + [1]
    offsets = (torch.arange(shape0, dtype=index.dtype, device=index.device) * shape1).view(new_shape)
    return (index + offsets).view(-1)


def _learned_scale_adjustment(scale, mlp, sq_dist, batch, n_sample_set, n_query_set, n_classes):
    """
    Learn a data-dependent adjustment of the scale factor used in the distance computation

    scale: float
        The original scale factor
    mlp: nn.Module
        The model used to learn the scale factor adjustment
    sq_dist: torch.Tensor, shape=(bs, n_total, n_total)
        A matrix of squared distances between all the examples
    n_sample_set: int
        Number of examples in the sample set
    n_query_set: int
        Number of examples in the query set
    n_classes: int
        Number of classes

    """
    global_mean = sq_dist.view(batch, -1).mean(1, keepdim=True).repeat(batch, sq_dist.size(1))
    global_std = sq_dist.view(batch, -1).std(1, keepdim=True).repeat(batch, sq_dist.size(1))
    avg_mean = sq_dist.mean(2).mean(1, keepdim=True).repeat(batch, sq_dist.size(1))
    avg_std = sq_dist.std(2).mean(1, keepdim=True).repeat(batch, sq_dist.size(1))
    sample_distances = sq_dist[:, :, :(n_classes * n_sample_set)].clone().view(batch, -1, n_sample_set)
    class_mean = sample_distances.mean(2).mean(1, keepdim=True).repeat(batch, sq_dist.size(1))
    class_std = sample_distances.std(2).mean(1, keepdim=True).repeat(batch, sq_dist.size(1))
    sample_mean = sq_dist.mean(2)
    sample_std = sq_dist.std(2)
    n = torch.ones_like(sample_mean) * n_classes
    k = torch.ones_like(sample_mean) * n_sample_set
    q = torch.ones_like(sample_mean) * n_query_set
    vin = torch.stack(
        [global_mean, global_std, sample_mean, sample_std, avg_mean, avg_std, class_mean, class_std, n, k, q], 2)
    scales = mlp(vin.view(-1, 11)).view(sq_dist.size(0), sq_dist.size(1), 1)
    return scale + F.softplus(scales)


def label_prop(sample_set, query_set, unlabeled_set=None, alpha=0.1, scale_factor=1, label_offset=0, apply_log=False,
               topk=0, method="global_consistancy", epsilon=1e-8, mlp=None, return_all=False, normalize_weights=False,
               debug_plot_path=None, propagator=None, labels=None, weights=None):
    """Uses the laplacian graph to smooth the labels matrix by "propagating" labels.

    Args:
        sample_set: Tensor of shape (batch, n_classes, n_sample_per_classes, z_dim) containing the representation z of
            each images.
        query_set: Tensor of shape (batch, n_classes, n_query_per_classes, z_dim) containing the representation z of
            each images.
        unlabeled_set: Tensor of shape (batch, n_classes, n_unlabeled_per_classes, z_dim) containing the representation
            z of each images.
        alpha: Smoothing factor in the laplacian graph
        scale_factor: scale modifying the euclidean distance before exponential kernel.
        label_offset: Applies an offset to the labels before propagating. Has an effect on the degree of uncertainty
            when apply_log is True.
        apply_log: if True, it is assumed that the label propagation methods returns un-normalized probabilities. Hence
            to return logits, applying logarithm is necessary.
        topk: limit the weight matrix to the topk most similiar
        method: "regularized_laplacian" or "global_consistancy".
        epsilon: small value used when apply_log is True.
        mlp: If 1, it trains an MLP to predict the scaling factor from weight matrix stats. If 2, it does it without
             passing backprop to the main architecture.
        return_all: For debugging purpose.

    Returns:
        Tensor of shape (batch, n_total_query, n_classes) representing the logits of each classes

    """
    init_query_size = query_set.size()[2]
    if unlabeled_set is not None:
        init_unlabel_size = unlabeled_set.size()[2]
        query_set = torch.cat((unlabeled_set, query_set), dim=2)  # XXX: Concat order is important to logit extraction

    # Get data shape
    batch, n_classes, n_query_pc, z_dim = query_set.size()
    batch, n_classes, n_sample_pc, z_dim = sample_set.size()

    if propagator is None:
        # Compute the pairwise distance between the examples of the sample and query sets
        # XXX: labels are set to a constant for the query set
        sq_dist, labels = pw_sq_dist(sample_set, query_set, label_offset)

        # Learn to adjust the scale
        if mlp is not None:
            scale_factor = _learned_scale_adjustment(scale_factor, mlp, sq_dist, batch, n_sample_pc, n_query_pc,
                                                     n_classes)

        # Compute similarity between the examples -- inversely proportional to distance
        weights = torch.exp(-0.5 * sq_dist / scale_factor ** 2)

        # Discard similarity for examples other than the top k most similar
        if topk > 0:
            weights, ind = torch.sort(weights, dim=2)  # ascending order
            weights[:, :, :-int(topk + 1)] *= 0
            weights = weights.scatter(2, ind, weights)

        # Normalize the weights
        if normalize_weights:
            weights = weights / torch.sum(weights, dim=2, keepdim=True)

        if (method == "regularized_laplacian") or (method is None):
            logits, propagator = regularized_laplacian(weights, labels, alpha=alpha)
        elif method == "global_consistancy":
            logits, propagator = global_consistency(weights, labels, alpha=alpha)
        else:
            raise Exception("Unknonwn method %s." % method)
    else:
        logits = _propagate(labels, propagator)

    if debug_plot_path is not None:
        print("Sample set:", n_sample_pc, "   Query set:", init_query_size,
              "   unlabeled set:", 0 if unlabeled_set is None else n_unlabeled_pc)
        # XXX: Only saves the first batch elements
        np.save(debug_plot_path + "_weights.npy", weights[0].detach().cpu().numpy())
        np.save(debug_plot_path + "_propagator.npy", propagator[0].detach().cpu().numpy())
        plt_labels = \
            np.argmax(torch.cat((_make_aligned_labels(sample_set),
                                 _make_aligned_labels(query_set)), dim=2).view(batch, -1, n_classes).cpu().numpy(),
                      axis=-1)[0]
        np.save(debug_plot_path + "_labels.npy", plt_labels)

    if apply_log:
        logits = torch.log(logits + epsilon)

    logits = logits.reshape(batch, n_classes, -1, n_classes)
    if return_all:
        # Extracts only the logits for the query set
        # TODO: verify that this works as expected.  <<<------- *****
        query_labels = _make_aligned_labels(query_set)
        query_logits = logits[:, :, -init_query_size:, :].reshape(batch, -1, n_classes)
        if unlabeled_set is not None:
            unlabeled_labels = query_labels[:, :, :init_unlabel_size, :]
            query_labels = query_labels[:, :, init_unlabel_size:, :]
            unlabel_logits = logits[:, :, :init_unlabel_size, :].reshape(batch, -1, n_classes)
        else:
            unlabel_logits = None
            unlabeled_labels = None
        return query_logits, unlabel_logits, labels, weights, _make_aligned_labels(
            sample_set), query_labels, unlabeled_labels, propagator
    else:
        # Extracts only the logits for the query set
        # TODO: verify that this works as expected.  <<<------- *****
        logits = logits.reshape(batch, n_classes, -1, n_classes)[:, :, -init_query_size:, :].reshape(batch, -1,
                                                                                                     n_classes)
        return logits

def make_one_hot_labels(labels, label_offset=0.):
    n_classes = labels.max() + 1
    assert (n_classes.item() == 5)
    to_one_hot = torch.eye(n_classes, device=labels.device, dtype=torch.float)
    one_hot_labels = to_one_hot[labels]
    mask = labels == -1
    one_hot_labels[mask, :] = label_offset
    return one_hot_labels


# def labelprop(suppot_set, query_set, suppot_labels, 
#               gaussian_scale=1, alpha=1,
#               propagator=None, weights=None, return_all=False,
#               apply_log=False, scale_bound="", standarize="", kernel="", square_root=False,
#               offset=0, epsilon=1e-6, dropout=0, n_classes=5):
#     if scale_bound == "softplus":
#         gaussian_scale = 0.01 + F.softplus(gaussian_scale)
#         alpha = 0.1 + F.softplus(alpha)
#     elif scale_bound == "square":
#         gaussian_scale = 1e-4 + gaussian_scale ** 2
#         alpha = 0.1 + alpha ** 2
#     elif scale_bound == "convex_relu":
#         #gaussian_scale = gaussian_scale ** 2
#         alpha = F.relu(alpha) + 0.1
#     elif scale_bound == "convex_square":
#         # gaussian_scale = gaussian_scale ** 2
#         alpha = 0.1 + alpha ** 2
#     elif scale_bound == "relu":
#         gaussian_scale = F.relu(gaussian_scale) + 0.01
#         alpha = F.relu(alpha) + 0.1
#     elif scale_bound == "constant":
#         gaussian_scale = 1
#         alpha = 1
#     elif scale_bound == "alpha_square":
#         alpha = 0.1 + F.relu(alpha)

#     samples = []
#     labels = []
#     offsets = []
#     is_labeled = []
#     for data, _labels in sets:
#         b, _, sample_size, c = data.size()
#         n = n_classes
#         samples.append(data)
#         offsets.append(sample_size)
#         if isinstance(_labels, bool):
#             labels.append(_make_aligned_labels(data))
#             if not (_labels):
#                 labels[-1].data[:] = labels[-1].data * 0 + offset
#             is_labeled.append(_labels)
#         else:
#             labels.append(_labels)
#             is_labeled.append((_labels.cpu().numpy() > 0).any())
#     samples = torch.cat(samples, dim=2)
#     labels = torch.cat(labels, dim=2).view(b, -1, n)

#     if propagator is None:
#         # Compute the pairwise distance between the examples of the sample and query sets
#         # XXX: labels are set to a constant for the query set
#         sq_dist = generalized_pw_sq_dist(samples, "euclidean")
#         if square_root:
#             sq_dist = (sq_dist + epsilon).sqrt()
#         if standarize == "all":
#             mask = sq_dist != 0
#          #   sq_dist = sq_dist - sq_dist[mask].mean()
#             sq_dist = sq_dist / sq_dist[mask].std()
#         elif standarize == "median":
#             mask = sq_dist != 0
#             gaussian_scale = torch.sqrt(
#                 0.5 * torch.median(sq_dist[mask]) / torch.log(torch.ones(1, device=sq_dist.device) + sq_dist.size(1)))
#         elif standarize == "frobenius":
#             mask = sq_dist != 0
#             sq_dist = sq_dist / (sq_dist[mask] ** 2).sum().sqrt()
#         elif standarize == "percentile":
#             mask = sq_dist != 2
#             sorted, indices = torch.sort(sq_dist.data[mask])
#             total = sorted.size(0)
#             gaussian_scale = sorted[int(total * 0.1)].detach()
#         if kernel == "rbf":
#             weights = torch.exp(-sq_dist * gaussian_scale)
#         elif kernel == "convex_rbf":
#             scales = torch.linspace(0.1, 10, gaussian_scale.size(0), device=sq_dist.device, dtype=sq_dist.dtype)
#             weights = torch.exp(-sq_dist.unsqueeze(1) * scales.view(1, -1, 1, 1))
#             weights = (weights * F.softmax(gaussian_scale.view(1, -1, 1, 1), dim=1)).sum(1)
#             # checknan(timessoftmax=weights)
#         elif kernel == "euclidean":
#             # Compute similarity between the examples -- inversely proportional to distance
#             weights = 1 / (gaussian_scale + sq_dist)
#         elif kernel == "softmax":
#             weights = F.softmax(-sq_dist / gaussian_scale, -1)

#         mask = (torch.eye(weights.size(1), dtype=weights.dtype, device=weights.device)[None, :, :]).view(1, weights.size(1), weights.size(2)).expand(weights.size(0), -1, -1)
#         weights = weights * (1 - mask)
#         # checknan(masking=weights)


#         logits, propagator = global_consistency(weights, labels, alpha=alpha)
#     else:
#         logits = _propagate(labels, propagator)

#     if apply_log:
#         logits = torch.log(logits + epsilon)

#     logits = logits.view(b, n, -1, n)
#     logits_ret = []
#     start = 0
#     for i, offset in enumerate(offsets):
#         logits_ret.append(logits[:, :, start:(start + offset), :].contiguous().view(b, -1, n))
#         start += offset
#     if return_all:
#         # Extracts only the logits for the query set
#         # TODO: verify that this works as expected.  <<<------- *****
#         labels = labels.view(b, n, -1, n)
#         labels_ret = []
#         start = 0
#         for i, offset in enumerate(offsets):
#             labels_ret.append(labels[:, :, start:(start + offset), :].contiguous().view(b, -1, n))
#             start += offset
#         return tuple(logits_ret + labels_ret + [weights, propagator, labels])
#     else:
#         # Extracts only the logits for the query set
#         # TODO: verify that this works as expected.  <<<------- *****
#         return tuple(logits_ret)

def standarized_label_prop(*sets, gaussian_scale=1, alpha=1,
                           propagator=None, weights=None, return_all=False,
                           apply_log=False, scale_bound="", standarize="", kernel="", square_root=False,
                           offset=0, epsilon=1e-6, dropout=0, n_classes=5):
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

    samples = []
    labels = []
    offsets = []
    is_labeled = []
    for data, _labels in sets:
        b, _, sample_size, c = data.size()
        n = n_classes
        samples.append(data)
        offsets.append(sample_size)
        if isinstance(_labels, bool):
            labels.append(_make_aligned_labels(data))
            if not (_labels):
                labels[-1].data[:] = labels[-1].data * 0 + offset
            is_labeled.append(_labels)
        else:
            labels.append(_labels)
            is_labeled.append((_labels.cpu().numpy() > 0).any())
    samples = torch.cat(samples, dim=2)
    labels = torch.cat(labels, dim=2).view(b, -1, n)

    if propagator is None:
        # Compute the pairwise distance between the examples of the sample and query sets
        # XXX: labels are set to a constant for the query set
        sq_dist = generalized_pw_sq_dist(samples, "euclidean")
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

        mask = (torch.eye(weights.size(1), dtype=weights.dtype, device=weights.device)[None, :, :]).view(1, weights.size(1), weights.size(2)).expand(weights.size(0), -1, -1)
        weights = weights * (1 - mask)
        # checknan(masking=weights)


        logits, propagator = global_consistency(weights, labels, alpha=alpha)
    else:
        logits = _propagate(labels, propagator)

    if apply_log:
        logits = torch.log(logits + epsilon)

    logits = logits.view(b, n, -1, n)
    logits_ret = []
    start = 0
    for i, offset in enumerate(offsets):
        logits_ret.append(logits[:, :, start:(start + offset), :].contiguous().view(b, -1, n))
        start += offset
    if return_all:
        # Extracts only the logits for the query set
        # TODO: verify that this works as expected.  <<<------- *****
        labels = labels.view(b, n, -1, n)
        labels_ret = []
        start = 0
        for i, offset in enumerate(offsets):
            labels_ret.append(labels[:, :, start:(start + offset), :].contiguous().view(b, -1, n))
            start += offset
        return tuple(logits_ret + labels_ret + [weights, propagator, labels])
    else:
        # Extracts only the logits for the query set
        # TODO: verify that this works as expected.  <<<------- *****
        return tuple(logits_ret)


def regularized_laplacian(weights, labels, alpha):
    """Uses the laplacian graph to smooth the labels matrix by "propagating" labels

    Args:
        weights: Tensor of shape (batch, n, n)
        labels: Tensor of shape (batch, n, n_classes)
        alpha: Scaler, acts as a smoothing factor
        apply_log: if True, it is assumed that the label propagation methods returns un-normalized probabilities. Hence
            to return logits, applying logarithm is necessary.
        epsilon: value added before applying log
    Returns:
        Tensor of shape (batch, n, n_classes) representing the logits of each classes
    """
    n = weights.shape[1]
    diag = torch.diag_embed(torch.sum(weights, dim=2))
    laplacian = diag - weights
    identity = torch.eye(n, dtype=laplacian.dtype, device=laplacian.device)[None, :, :]
    propagator = torch.inverse(identity + alpha * laplacian)

    return _propagate(labels, propagator), propagator


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
    alpha_ = 1 / (1 + alpha)
    beta_ = alpha / (1 + alpha)
    n = weights.shape[1]
    identity = torch.eye(n, dtype=weights.dtype, device=weights.device)[None, :, :]
    #weights = weights * (1. - identity)  # zero out diagonal
    isqrt_diag = 1. / torch.sqrt(1e-4 + torch.sum(weights, dim=2))
    # checknan(laplacian=isqrt_diag)
    S = weights * isqrt_diag[:, None, :] * isqrt_diag[:, :, None]
    # checknan(normalizedlaplacian=S)
    propagator = identity - alpha_ * S
    propagator = torch.inverse(propagator) * beta_
    # checknan(propagator=propagator)

    return _propagate(labels, propagator, scaling=1), propagator


def _propagate(labels, propagator, scaling=1.):
    return torch.matmul(propagator, labels) * scaling


class MLP(torch.nn.Module):
    def __init__(self, detach=False):
        super().__init__()
        self.__detach = detach
        self.linear1 = torch.nn.Linear(11, 128)
        self.linear2 = torch.nn.Linear(128, 16)
        self.linear3 = torch.nn.Linear(16, 1)

    def forward(self, x):
        if self.__detach:
            x = x.detach()
        x = F.relu(self.linear1(x), True)
        x = F.relu(self.linear2(x), True)
        return self.linear3(x)

class Distance(torch.nn.Module):
    def __init__(self, exp_params):
        """ Helper to obtain a distance function from a string
        Args:
            distance_type: string indicating the distance type
        """
        super().__init__()
        self.exp_params = exp_params
        self.distance_type, *args = exp_params["distance_type"].split(',')
        if self.distance_type in ["euclidean", "prototypical"]:
            self.d = prototype_distance
        elif self.distance_type == "labelprop":
            self.register_buffer("moving_alpha", torch.ones(1) * self.exp_params["labelprop_alpha_prior"])
            if self.exp_params["kernel_type"] == "convex_rbf":
                scale_size = 10
            else:
                scale_size = 1
            self.register_buffer("moving_gaussian_scale", torch.ones(scale_size) * exp_params["labelprop_scale_prior"])
            self.d = partial(standarized_label_prop, scale_bound=self.exp_params["kernel_bound"],
                             kernel=self.exp_params["kernel_type"],
                             standarize=self.exp_params["kernel_standarization"],
                             square_root=self.exp_params["kernel_square_root"],
                             apply_log=True)
        elif self.distance_type == "matching":
            matching_distance, = args
            self.d = partial(matching_nets, distance_type=matching_distance)
        elif self.distance_type == "labelprop_boris":
            """
                TODO(@boris) define here the relation network, and define a new "label_prop" above, alike the original 
                one parameters inside Distance are taken into account by the optimizer
            """
            self.d = partial(label_prop, apply_log=True)

    def forward(self, *sets, **kwargs):
        return self.d(*sets, **kwargs)