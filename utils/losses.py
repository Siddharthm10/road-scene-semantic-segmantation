import torch
import torch.nn as nn

def focal_loss(inputs, targets, alpha=0.25, gamma=2):
    ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss_val = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss_val.mean()

def dice_loss(inputs, targets, smooth=1):
    num_classes = inputs.shape[1]
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    inputs_soft = torch.softmax(inputs, dim=1)
    intersection = (inputs_soft * targets_one_hot).sum(dim=(2,3))
    union = inputs_soft.sum(dim=(2,3)) + targets_one_hot.sum(dim=(2,3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

class HybridLoss(nn.Module):
    def __init__(self, weight_focal=0.5, weight_dice=0.5, alpha=0.25, gamma=2):
        super(HybridLoss, self).__init__()
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        loss_focal = focal_loss(inputs, targets, alpha=self.alpha, gamma=self.gamma)
        loss_dice = dice_loss(inputs, targets)
        return self.weight_focal * loss_focal + self.weight_dice * loss_dice
