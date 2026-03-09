import numpy as np
import torch
from torchvision.datasets import CIFAR10, CIFAR100
from collections import Counter

class corrupted_CIFAR10(CIFAR10):
    def __init__(self, root, noise_rho=0.0, rand_number=0, train=True, transform=None, target_transform=None, download=False):
        super(corrupted_CIFAR10, self).__init__(root=root, train=train, transform=transform,
                                                target_transform=target_transform, download=download)

        self.noise_rho = noise_rho
        self.rand_number = rand_number

        if self.train and self.noise_rho > 0:
            self._inject_asymmetric_noise()

    def _inject_asymmetric_noise(self):
        np.random.seed(self.rand_number)
        num_classes = 10
        targets = np.array(self.targets)

        # Define asymmetric mapping: class A → B
        transition = {9: 1, 2: 0, 3: 5, 5: 3}  # truck→automobile, bird→airplane, cat↔dog
        noisy_idx = np.random.choice(len(targets), int(self.noise_rho * len(targets)), replace=False)

        for idx in noisy_idx:
            label = targets[idx]
            if label in transition:
                targets[idx] = transition[label]

        self.targets = targets.tolist()

    def get_cls_num_list(self):
        counter = Counter(self.targets)
        cls_num_list = [counter.get(i, 0) for i in range(10)]
        return cls_num_list

class corrupted_CIFAR100(CIFAR100):
    def __init__(self, root, noise_rho=0.0, rand_number=0, train=True, transform=None, target_transform=None, download=False):
        super(corrupted_CIFAR100, self).__init__(root=root, train=train, transform=transform,
                                                 target_transform=target_transform, download=download)

        self.noise_rho = noise_rho
        self.rand_number = rand_number

        self.fine_to_coarse = self._build_fine_to_coarse()

        self.noise_or_not = np.zeros(len(self.targets), dtype=int)

        if self.train and self.noise_rho > 0:
            self._inject_asymmetric_noise()
    
    def _build_fine_to_coarse(self):
        """
        Build a fine_label_id → coarse_label_id mapping
        Fully compatible with torchvision.datasets.CIFAR100.classes
        """

        # Coarse groups: list of lists of fine label indices (0–99)
        coarse_groups = [
            [0, 1, 2, 3, 4],       # aquatic mammals
            [5, 6, 7, 8, 9],       # fish
            [10, 11, 12, 13, 14],  # flowers
            [15, 16, 17, 18, 19],  # food containers
            [20, 21, 22, 23, 24],  # fruit and vegetables
            [25, 26, 27, 28, 29],  # household electrical devices
            [30, 31, 32, 33, 34],  # household furniture
            [35, 36, 37, 38, 39],  # insects
            [40, 41, 42, 43, 44],  # large carnivores
            [45, 46, 47, 48, 49],  # large man-made outdoor things
            [50, 51, 52, 53, 54],  # large natural outdoor scenes
            [55, 56, 57, 58, 59],  # large omnivores and herbivores
            [60, 61, 62, 63, 64],  # medium-sized mammals
            [65, 66, 67, 68, 69],  # non-insect invertebrates
            [70, 71, 72, 73, 74],  # people
            [75, 76, 77, 78, 79],  # reptiles
            [80, 81, 82, 83, 84],  # small mammals
            [85, 86, 87, 88, 89],  # trees
            [90, 91, 92, 93, 94],  # vehicles 1
            [95, 96, 97, 98, 99],  # vehicles 2
        ]

        # Construct index-based mapping
        fine_to_coarse = [None] * 100
        for coarse_id, fine_list in enumerate(coarse_groups):
            for fine_id in fine_list:
                fine_to_coarse[fine_id] = coarse_id

        return fine_to_coarse

    def _inject_asymmetric_noise(self):
        np.random.seed(self.rand_number)
        targets = np.array(self.targets)
        coarse_targets = np.array([self.fine_to_coarse[fine_id] for fine_id in self.targets])

        group_dict = {}
        for idx, coarse in enumerate(coarse_targets):
            if coarse not in group_dict:
                group_dict[coarse] = []
            group_dict[coarse].append(idx)

        noisy_idx = np.random.choice(len(targets), int(self.noise_rho * len(targets)), replace=False)
        for idx in noisy_idx:
            label = targets[idx]
            coarse = coarse_targets[idx]
            same_group = group_dict[coarse]
            candidate_labels = [targets[i] for i in same_group if targets[i] != label]
            if candidate_labels:
                targets[idx] = np.random.choice(candidate_labels)
                self.noise_or_not[idx] = 1

        self.targets = targets.tolist()
    
    def get_cls_num_list(self):
        counter = Counter(self.targets)
        cls_num_list = [counter.get(i, 0) for i in range(100)]
        return cls_num_list

    def get_noisy_indices(self):
        """Return a list of indices that were corrupted."""
        return np.where(self.noise_or_not == 1)[0].tolist()

    def __getitem__(self, index):
        """Return (image, label, index) for tracking global index."""
        image, label = super().__getitem__(index)
        return image, label, index
