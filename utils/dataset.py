import os
import numpy as np
import torch
from torchvision.datasets import Cityscapes
from utils.map_trainId_imgs import convert_id_to_train_id

class CityscapesDatasetWrapper(Cityscapes):
    def __init__(self, root, split, mode, target_type, joint_transform=None):
        super().__init__(root, split=split, mode=mode, target_type=target_type)
        self.joint_transform = joint_transform

    def __getitem__(self, index):
        image, target = super().__getitem__(index)

        # Only keep one label target if multiple are returned
        if isinstance(target, (list, tuple)):
            target = target[0]

        # Apply joint transformations (e.g., resize, crop, flip)
        if self.joint_transform is not None:
            image, target = self.joint_transform(image, target)

        # Convert raw labels to train IDs
        target = convert_id_to_train_id(target)

        return image, target
