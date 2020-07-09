import sys
import torchvision
import torch
from torch.utils.data import Dataset
import json
import os
import numpy as np
from PIL import Image

class NonEpisodicCUB(Dataset):
    name="CUB"
    task="cls"
    split_paths = {"train":"train", "test":"test", "valid": "val"}
    c = 3
    h = 84
    w = 84

    def __init__(self, data_root, split, transforms, rotation_labels=[0, 1, 2, 3], **kwargs):
        """ Constructor

        Args:
            split: data split
            few_shot_sampler: FewShotSampler instance
            task: dataset task (if more than one)
            size: number of tasks to generate (int)
            disjoint: whether to create disjoint splits.
        """
        self.data_root = data_root
        self.split = {"train":"base", "val":"val", "valid":"val", "test":"novel"}[split]
        with open(os.path.join(self.data_root, "few_shot_lists", "%s.json" %self.split), 'r') as infile:
            self.metadata = json.load(infile)
        self.transforms = transforms
        self.rotation_labels = rotation_labels
        self.labels = np.array(self.metadata['image_labels'])
        label_map = {l: i for i, l in enumerate(sorted(np.unique(self.labels)))}
        self.labels = np.array([label_map[l] for l in self.labels])
        self.size = len(self.metadata["image_labels"])

    def next_run(self):
        pass

    def rotate_img(self, img, rot):
        if rot == 0:  # 0 degrees rotation
            return img
        elif rot == 90:  # 90 degrees rotation
            return np.flipud(np.transpose(img, (1, 0, 2)))
        elif rot == 180:  # 90 degrees rotation
            return np.fliplr(np.flipud(img))
        elif rot == 270:  # 270 degrees rotation / or -90
            return np.transpose(np.flipud(img), (1, 0, 2))
        else:
            raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

    def __getitem__(self, item):
        image = np.array(Image.open(self.metadata["image_names"][item]).convert("RGB"))
        images = self.transforms(image) * 2 - 1
        return images, int(self.labels[item])

    def __len__(self):
        return len(self.labels)

class RotatedNonEpisodicCUB(NonEpisodicCUB):
    name="CUB"
    task="cls"
    split_paths = {"train":"train", "test":"test", "valid": "val"}
    c = 3
    h = 84
    w = 84

    def __init__(self, *args, **kwargs):
        """ Constructor

        Args:
            split: data split
            few_shot_sampler: FewShotSampler instance
            task: dataset task (if more than one)
            size: number of tasks to generate (int)
            disjoint: whether to create disjoint splits.
        """
        super().__init__(*args, **kwargs)

    def rotate_img(self, img, rot):
        if rot == 0:  # 0 degrees rotation
            return img
        elif rot == 90:  # 90 degrees rotation
            return np.flipud(np.transpose(img, (1, 0, 2)))
        elif rot == 180:  # 90 degrees rotation
            return np.fliplr(np.flipud(img))
        elif rot == 270:  # 270 degrees rotation / or -90
            return np.transpose(np.flipud(img), (1, 0, 2))
        else:
            raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

    def __getitem__(self, item):
        image = np.array(Image.open(self.metadata["image_names"][item]).convert("RGB"))
        if np.random.randint(2):
            image = np.fliplr(image)
        image_90 = self.transforms(self.rotate_img(image, 90))
        image_180 = self.transforms(self.rotate_img(image, 180))
        image_270 = self.transforms(self.rotate_img(image, 270))
        images = torch.stack([self.transforms(image), image_90, image_180, image_270]) * 2 - 1
        return images, torch.ones(4, dtype=torch.long)*int(self.labels[item]), torch.LongTensor(self.rotation_labels)