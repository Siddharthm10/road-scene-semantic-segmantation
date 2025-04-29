import numpy as np

def compute_mIoU(pred, target, num_classes):
    """
    pred and target are both HxW or NxHxW (i.e. per-image or per-batch).
    Here, for simplicity, assume we've already aggregated them as NxHxW
    or we do it inside the for loop. The key idea is we skip pixels
    labeled 255 in the ground truth.
    """
    ious = []
    for cls in range(num_classes):
        # Create a valid_mask that excludes 'ignore' pixels
        valid_mask = (target != 255)

        # pred == cls, but only in valid pixels
        pred_inds = (pred == cls) & valid_mask
        # target == cls, but only in valid pixels
        target_inds = (target == cls) & valid_mask

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            iou = float('nan')  # or skip
        else:
            iou = intersection / union
        ious.append(iou)

    # Filter out any NaNs (e.g., if no pixels of that class appeared)
    ious_clean = [iou for iou in ious if not np.isnan(iou)]
    mIoU = np.mean(ious_clean) if len(ious_clean) > 0 else 0

    return ious, mIoU
