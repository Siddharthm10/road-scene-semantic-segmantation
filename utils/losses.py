import torch
import torch.nn as nn

def focal_loss(inputs, targets, alpha=0.5, gamma=1.5, ignore_index=255):
    ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', ignore_index=ignore_index)
    pt = torch.exp(-ce_loss)
    focal_loss_val = alpha * (1 - pt) ** gamma * ce_loss
    valid_mask = (targets != ignore_index).float()
    if valid_mask.sum() > 0:
        return (focal_loss_val * valid_mask).sum() / valid_mask.sum()
    else:
        return focal_loss_val.mean()

def dice_loss(inputs, targets, num_classes, smooth=0.1, ignore_index=255):
    num_classes = inputs.shape[1]
    valid_mask = (targets != ignore_index)
    targets_clamped = targets.clone()
    targets_clamped[~valid_mask] = 0
    targets_clamped = targets_clamped.long()
    targets_one_hot = torch.nn.functional.one_hot(targets_clamped, num_classes).permute(0, 3, 1, 2).float()
    inputs_soft = torch.softmax(inputs, dim=1)
    valid_mask = valid_mask.unsqueeze(1)  # Shape: (B, 1, H, W)
    targets_one_hot *= valid_mask
    intersection = (inputs_soft * targets_one_hot).sum(dim=(2, 3))
    union = inputs_soft.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


class HybridLoss(nn.Module):
    def __init__(self, weight_focal=0.7, weight_dice=0.3, alpha=0.5, gamma=1.5, smooth=0.1, ignore_index=255, num_classes=None):
        super(HybridLoss, self).__init__()
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        loss_focal = focal_loss(inputs, targets, alpha=self.alpha, gamma=self.gamma, ignore_index=self.ignore_index)
        loss_dice = dice_loss(inputs, targets, num_classes=self.num_classes, smooth=self.smooth, ignore_index=self.ignore_index)
        return self.weight_focal * loss_focal + self.weight_dice * loss_dice
