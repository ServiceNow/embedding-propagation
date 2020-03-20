import numpy as np
import torch
import torchvision
import os
from src.datasets.episodic_dataset import EpisodicDataset, FewShotSampler
import pickle as pkl

# Inherit order is important, FewShotDataset constructor is prioritary
class EpisodicMiniImagenet(EpisodicDataset):
    tasks_type = "clss"
    name = "miniimagenet"
    episodic=True
    split_paths = {"train":"train", "valid":"val", "val":"val", "test": "test"}
    # c = 3
    # h = 84
    # w = 84

    def __init__(self, data_root, split, sampler, size, transforms):
        """ Constructor

        Args:
            split: data split
            few_shot_sampler: FewShotSampler instance
            task: dataset task (if more than one)
            size: number of tasks to generate (int)
            disjoint: whether to create disjoint splits.
        """
        self.data_root = os.path.join(data_root, "mini-imagenet-%s.npz")
        self.split = split
        data = np.load(self.data_root % self.split_paths[split])
        self.features = data["features"]
        labels = data["targets"]
        del(data)
        super().__init__(labels, sampler, size, transforms)

    def sample_images(self, indices):
        return self.features[indices]

    def __iter__(self):
        return super().__iter__()

# Inherit order is important, FewShotDataset constructor is prioritary
class EpisodicMiniImagenetPkl(EpisodicDataset):
    tasks_type = "clss"
    name = "miniimagenet"
    episodic=True
    split_paths = {"train":"train", "valid":"val", "val":"val", "test": "test"}
    # c = 3
    # h = 84
    # w = 84

    def __init__(self, data_root, split, sampler, size, transforms):
        """ Constructor

        Args:
            split: data split
            few_shot_sampler: FewShotSampler instance
            task: dataset task (if more than one)
            size: number of tasks to generate (int)
            disjoint: whether to create disjoint splits.
        """
        self.data_root = os.path.join(data_root, "mini-imagenet-cache-%s.pkl")
        self.split = split
        with open(self.data_root % self.split_paths[split], 'rb') as infile:
            data = pkl.load(infile)
        self.features = data["image_data"]
        label_names = data["class_dict"].keys()
        labels = np.zeros((self.features.shape[0],), dtype=int)
        for i, name in enumerate(sorted(label_names)):
            labels[np.array(data['class_dict'][name])] = i
        del(data)
        super().__init__(labels, sampler, size, transforms)

    def sample_images(self, indices):
        return self.features[indices]

    def __iter__(self):
        return super().__iter__()

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from src.tools.plot_episode import plot_episode
    import time
    sampler = FewShotSampler(5, 5, 15, 0)
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                 torchvision.transforms.ToTensor(),
                                                ])
    dataset = EpisodicMiniImagenetPkl('./miniimagenet', 'train', sampler, 1000, transforms)
    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x)
    for batch in loader:
        print(np.unique(batch[0]["targets"].view(20, 5).numpy()))
        # plot_episode(batch[0], classes_first=False)
        # time.sleep(1)

