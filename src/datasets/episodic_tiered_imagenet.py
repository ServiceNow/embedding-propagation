import sys
import torchvision
import torch
from torch.utils.data import Dataset
from .episodic_dataset import EpisodicDataset, FewShotSampler
import json
import os
import numpy as np
import numpy
import cv2
import pickle as pkl

# Inherit order is important, FewShotDataset constructor is prioritary
class EpisodicTieredImagenet(EpisodicDataset):
    tasks_type = "clss"
    name = "tiered-imagenet"
    split_paths = {"train":"train", "test":"test", "valid": "val"}
    c = 3
    h = 84
    w = 84
    def __init__(self, data_root, split, sampler, size, transforms):
        self.data_root = data_root
        self.split = split
        img_path = os.path.join(self.data_root, "%s_images_png.pkl" %(split))
        label_path = os.path.join(self.data_root, "%s_labels.pkl" %(split))
        with open(img_path, 'rb') as infile:
            self.features = pkl.load(infile, encoding="bytes")
        with open(label_path, 'rb') as infile:
            labels = pkl.load(infile, encoding="bytes")[b'label_specific']
        super().__init__(labels, sampler, size, transforms)
    
    def sample_images(self, indices):
        return [cv2.imdecode(self.features[i], cv2.IMREAD_COLOR)[:,:,::-1] for i in indices]

    def __iter__(self):
        return super().__iter__()

if __name__ == '__main__':
    import sys
    from torch.utils.data import DataLoader
    from tools.plot_episode import plot_episode
    sampler = FewShotSampler(5, 5, 15, 0)
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                 torchvision.transforms.Resize((84,84)),
                                                 torchvision.transforms.ToTensor(),
                                                ])
    dataset = EpisodicTieredImagenet("train", sampler, 10, transforms)
    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x)
    for batch in loader:
        plot_episode(batch[0], classes_first=False)
