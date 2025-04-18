import torch
import torch.nn as nn

def focal_loss(inputs, targets, alpha=0.25, gamma=2, ignore_index=255):
    # Compute cross-entropy loss with ignore_index.
    ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', ignore_index=ignore_index)
    pt = torch.exp(-ce_loss)
    focal_loss_val = alpha * (1 - pt) ** gamma * ce_loss
    # Only average over non-ignored pixels.
    valid_mask = (targets != ignore_index).float()
    if valid_mask.sum() > 0:
        return (focal_loss_val * valid_mask).sum() / valid_mask.sum()
    else:
        return focal_loss_val.mean()

def dice_loss(inputs, targets, smooth=1, ignore_index=255):
    num_classes = inputs.shape[1]
    # Create a mask for valid pixels (those not equal to ignore_index)
    valid_mask = (targets != ignore_index)
    # Replace ignore pixels with a valid label (e.g., 0)
    targets_clamped = targets.clone()
    targets_clamped[~valid_mask] = 0
    # Clamp all values to be in the valid range [0, num_classes-1]
    targets_clamped = torch.clamp(targets_clamped, 0, num_classes - 1)
    # Convert to one-hot encoding
    targets_one_hot = torch.nn.functional.one_hot(targets_clamped, num_classes).permute(0, 3, 1, 2).float()
    inputs_soft = torch.softmax(inputs, dim=1)
    # Zero out contributions from ignored pixels
    valid_mask = valid_mask.unsqueeze(1)  # shape (B, 1, H, W)
    targets_one_hot = targets_one_hot * valid_mask
    intersection = (inputs_soft * targets_one_hot).sum(dim=(2,3))
    union = inputs_soft.sum(dim=(2,3)) + targets_one_hot.sum(dim=(2,3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

class HybridLoss(nn.Module):
    def __init__(self, weight_focal=0.5, weight_dice=0.5, alpha=0.25, gamma=2, ignore_index=255):
        super(HybridLoss, self).__init__()
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        loss_focal = focal_loss(inputs, targets, alpha=self.alpha, gamma=self.gamma, ignore_index=self.ignore_index)
        loss_dice = dice_loss(inputs, targets, ignore_index=self.ignore_index)
        return self.weight_focal * loss_focal + self.weight_dice * loss_dice
