import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torch

class SegmentationTrainTransform:
    def __init__(self, base_size=(1024, 512), scale_range=(1.0, 1.5), apply_color_jitter=True):
        self.base_size = base_size              # Target crop size after scaling (W, H)
        self.scale_range = scale_range
        self.apply_color_jitter = apply_color_jitter

        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )

    def __call__(self, image, target):
        # Random horizontal flip
        if random.random() > 0.5:
            image = F.hflip(image)
            target = F.hflip(target)

        # Random scaling up
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        new_w = int(self.base_size[0] * scale)
        new_h = int(self.base_size[1] * scale)

        image = F.resize(image, (new_h, new_w), interpolation=F.InterpolationMode.BILINEAR)
        target = F.resize(target, (new_h, new_w), interpolation=F.InterpolationMode.NEAREST)

        # Random crop back to 1024x512 (W x H)
        crop_h, crop_w = self.base_size[1], self.base_size[0]
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(crop_h, crop_w))
        image = F.crop(image, i, j, h, w)
        target = F.crop(target, i, j, h, w)

        # Optional: color jitter (image only)
        if self.apply_color_jitter:
            image = self.color_jitter(image)

        # Convert to tensor + normalize
        image = F.to_tensor(image)
        image = F.normalize(image, mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        target = torch.as_tensor(np.array(target), dtype=torch.long)

        return image, target


class SegmentationValTransform:
    def __init__(self, resize_to=(512, 1024)):  # H x W
        self.resize_to = resize_to

    def __call__(self, image, target):
        image = F.resize(image, self.resize_to, interpolation=F.InterpolationMode.BILINEAR)
        target = F.resize(target, self.resize_to, interpolation=F.InterpolationMode.NEAREST)

        image = F.to_tensor(image)
        image = F.normalize(image, mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        target = torch.as_tensor(np.array(target), dtype=torch.long)
        return image, target
