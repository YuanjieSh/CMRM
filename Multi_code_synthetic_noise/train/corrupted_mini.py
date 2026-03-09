import os
import json
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import math


class corrupted_MiniImageNet(Dataset):
    cls_num = 100

    def __init__(self,
                 root_dir: str,
                 csv_name: str,
                 json_path: str,
                 train=True,
                 noise_rho=0.0,
                 rand_number=0,
                 transform=None):
        images_dir = os.path.join(root_dir, "images")
        assert os.path.exists(images_dir), "dir:'{}' not found.".format(images_dir)
        assert os.path.exists(json_path), "file:'{}' not found.".format(json_path)

        self.label_dict = json.load(open(json_path, "r"))
        csv_path = os.path.join(root_dir, csv_name)
        assert os.path.exists(csv_path), "file:'{}' not found.".format(csv_path)

        csv_data = pd.read_csv(csv_path)
        self.img_paths = [os.path.join(images_dir, i) for i in csv_data["filename"].values]
        self.img_label = [self.label_dict[i][0] for i in csv_data["label"].values]
        self.targets = self.img_label.copy()
        self.samples = list(zip(self.img_paths, self.img_label))
        self.transform = transform
        self.total_num = len(self.samples)

        np.random.seed(rand_number)
        self.rand_number = rand_number
        self.noise_rho = noise_rho

        if train and self.noise_rho > 0:
            self._inject_asymmetric_noise()

    def _inject_asymmetric_noise(self):
        targets = np.array(self.targets)
        num_classes = self.cls_num
        noisy_idx = np.random.choice(len(targets), int(self.noise_rho * len(targets)), replace=False)

        # Simple asymmetric noise: label i → i+1 mod 100
        for idx in noisy_idx:
            original = targets[idx]
            new_label = (original + 1) % num_classes
            targets[idx] = new_label

        self.targets = targets.tolist()
        self.img_label = self.targets  # make sure __getitem__ uses noisy label
        self.samples = list(zip(self.img_paths, self.targets))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        img_path, label = self.samples[item]
        img = Image.open(img_path)
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(img_path))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

    def get_cls_num_list(self):
        from collections import Counter
        counter = Counter(self.targets)
        cls_num_list = [counter.get(i, 0) for i in range(self.cls_num)]
        return cls_num_list



