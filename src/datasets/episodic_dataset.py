import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import collections
from copy import deepcopy

_DataLoader = DataLoader
class EpisodicDataLoader(_DataLoader):
    def __iter__(self):
        if isinstance(self.dataset, EpisodicDataset):
            self.dataset.__iter__() 
        else:
            pass    
        return super().__iter__()
torch.utils.data.DataLoader = EpisodicDataLoader

class FewShotSampler():
    FewShotTask = collections.namedtuple("FewShotTask", ["nclasses", "support_size", "query_size", "unlabeled_size"])
    def __init__(self, nclasses, support_size, query_size, unlabeled_size):
        self.task = self.FewShotTask(nclasses, support_size, query_size, unlabeled_size)

    def sample(self):
        return deepcopy(self.task)

class EpisodicDataset(Dataset):
    def __init__(self, labels, sampler, size, transforms):
        self.labels = labels
        self.sampler = sampler
        self.labelset = np.unique(labels)
        self.indices = np.arange(len(labels))
        self.transforms = transforms
        self.reshuffle()
        self.size = size
    
    def reshuffle(self):
        """
        Helper method to randomize tasks again
        """
        self.clss_idx = [np.random.permutation(self.indices[self.labels == label]) for label in self.labelset]
        self.starts = np.zeros(len(self.clss_idx), dtype=int)
        self.lengths = np.array([len(x) for x in self.clss_idx])

    def gen_few_shot_task(self, nclasses, size):
        """ Iterates through the dataset sampling tasks

        Args:
            n: FewShotTask.n
            sample_size: FewShotTask.k
            query_size: FewShotTask.k (default), else query_set_size // FewShotTask.n

        Returns: Sampled task or None in the case the dataset has been exhausted.

        """
        classes = np.random.choice(self.labelset, nclasses, replace=False)
        starts = self.starts[classes]
        reminders = self.lengths[classes] - starts
        if np.min(reminders) < size:
            return None
        sample_indices = np.array(
            [self.clss_idx[classes[i]][starts[i]:(starts[i] + size)] for i in range(len(classes))])
        sample_indices = np.reshape(sample_indices, [nclasses, size]).transpose()
        self.starts[classes] += size
        return sample_indices.flatten()

    def sample_task_list(self):
        """ Generates a list of tasks (until the dataset is exhausted)

        Returns: the list of tasks [(FewShotTask object, task_indices), ...]

        """
        task_list = []
        task_info = self.sampler.sample()
        nclasses, support_size, query_size, unlabeled_size = task_info
        unlabeled_size = min(unlabeled_size, self.lengths.min() - support_size - query_size)
        task_info = FewShotSampler.FewShotTask(nclasses=nclasses,
                                                support_size=support_size, 
                                                query_size=query_size, 
                                                unlabeled_size=unlabeled_size)
        k = support_size + query_size + unlabeled_size
        if np.any(k > self.lengths):
            raise RuntimeError("Requested more samples than existing")
        few_shot_task = self.gen_few_shot_task(nclasses, k)

        while few_shot_task is not None:
            task_list.append((task_info, few_shot_task))
            task_info = self.sampler.sample()
            nclasses, support_size, query_size, unlabeled_size = task_info
            k = support_size + query_size + unlabeled_size
            few_shot_task = self.gen_few_shot_task(nclasses, k)
        return task_list

    def sample_images(self, indices):
        raise NotImplementedError

    def __getitem__(self, idx):
        """ Reads the idx th task (episode) from disk

        Args:
            idx: task index

        Returns: task dictionary with (dataset (char), task (char), dim (tuple), episode (Tensor))

        """
        fs_task_info, indices = self.task_list[idx]
        ordered_argindices = np.argsort(indices)
        ordered_indices = np.sort(indices)
        nclasses, support_size, query_size, unlabeled_size = fs_task_info
        k = support_size + query_size + unlabeled_size
        _images = self.sample_images(ordered_indices)
        images = torch.stack([self.transforms(_images[i]) for i in np.argsort(ordered_argindices)])
        total, c, h, w = images.size()
        assert(total == (k * nclasses))
        images = images.view(k, nclasses, c, h, w)
        del(_images)
        images = images * 2 - 1
        targets = np.zeros([nclasses * k], dtype=int)
        targets[ordered_argindices] = self.labels[ordered_indices, ...].ravel()
        sample = {"dataset": self.name,
                  "channels": c,
                  "height": h,
                  "width": w,
                  "nclasses": nclasses,
                  "support_size": support_size,
                  "query_size": query_size,
                  "unlabeled_size": unlabeled_size,
                  "targets": torch.from_numpy(targets),
                  "support_set": images[:support_size, ...],
                  "query_set": images[support_size:(support_size +
                                                   query_size), ...],
                  "unlabeled_set": None if unlabeled_size == 0 else images[(support_size + query_size):, ...]}
        return sample    


    def __iter__(self):
        # print("Prefetching new epoch episodes")
        self.task_list = []
        while len(self.task_list) < self.size:
            self.reshuffle()
            self.task_list += self.sample_task_list()
        # print("done prefetching.")
        return []

    def __len__(self):
        return self.size