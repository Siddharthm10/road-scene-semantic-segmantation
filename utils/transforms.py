import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torch

class SegmentationTrainTransform:
    def __init__(self, output_size=(256, 512), scale_range=(0.8, 1.2)):
        self.output_size = output_size
        self.scale_range = scale_range

    def __call__(self, image, target):
        # Random horizontal flip
        if random.random() > 0.5:
            image = F.hflip(image)
            target = F.hflip(target)

        # Resize both image and target to the desired output size directly
        # image = F.resize(image, self.output_size, interpolation=F.InterpolationMode.BILINEAR)
        # target = F.resize(target, self.output_size, interpolation=F.InterpolationMode.NEAREST)

        # Convert image to tensor and normalize
        image = F.to_tensor(image)
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Convert target to tensor
        target = torch.as_tensor(np.array(target), dtype=torch.long)

        return image, target

class SegmentationValTransform:
    def __init__(self, output_size=(256, 512)):
        self.output_size = output_size

    def __call__(self, image, target):
        # image = F.resize(image, self.output_size, interpolation=F.InterpolationMode.BILINEAR)
        # target = F.resize(target, self.output_size, interpolation=F.InterpolationMode.NEAREST)
        
        image = F.to_tensor(image)
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        target = torch.as_tensor(np.array(target), dtype=torch.long)

        return image, target
