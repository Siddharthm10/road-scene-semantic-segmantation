import numpy as np

def compute_mIoU(pred, target, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        ious.append(iou)
    ious_clean = [iou for iou in ious if not np.isnan(iou)]
    mIoU = np.mean(ious_clean) if ious_clean else 0
    return ious, mIoU
