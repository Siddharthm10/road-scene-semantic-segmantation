import os
import numpy as np
import torch
from torchvision.datasets import Cityscapes

class CityscapesDatasetWrapper(Cityscapes):
    def __init__(self, root, split, mode, target_type, joint_transform=None):
        super().__init__(root, split=split, mode=mode, target_type=target_type)
        self.joint_transform = joint_transform

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        if isinstance(target, (list, tuple)):
            target = target[0]
        if self.joint_transform is not None:
            image, target = self.joint_transform(image, target)
        return image, target
