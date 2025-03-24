import numpy as np

def compute_mIoU(preds, targets, num_classes, ignore_index=255):
    # First, mask out pixels that should be ignored
    mask = targets != ignore_index
    preds = preds[mask]
    targets = targets[mask]
    
    ious = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        ious.append(iou)
    
    # Compute mean IoU over classes with valid pixels
    ious_cleaned = [iou for iou in ious if not np.isnan(iou)]
    mIoU = np.mean(ious_cleaned) if ious_cleaned else 0
    return ious, mIoU
