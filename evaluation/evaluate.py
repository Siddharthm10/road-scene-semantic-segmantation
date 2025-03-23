#!/usr/bin/env python
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from utils.dataset import CityscapesDatasetWrapper
from utils.transforms import SegmentationValTransform
from utils.metrics import compute_mIoU
from models.unet import UNet

def evaluate_model(model, data_loader, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_ious = np.zeros(num_classes)
    count_ious = np.zeros(num_classes)
    criterion = nn.CrossEntropyLoss()  # Using cross-entropy for reporting loss
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            per_class_ious, _ = compute_mIoU(preds.cpu().numpy(), masks.cpu().numpy(), num_classes)
            for cls in range(num_classes):
                if not np.isnan(per_class_ious[cls]):
                    total_ious[cls] += per_class_ious[cls]
                    count_ious[cls] += 1
    avg_loss = total_loss / len(data_loader)
    avg_ious = [total_ious[i] / count_ious[i] if count_ious[i] > 0 else 0 for i in range(num_classes)]
    mIoU = np.mean(avg_ious)
    return avg_loss, mIoU, avg_ious

def main():
    parser = argparse.ArgumentParser(description="Evaluate U-Net on Cityscapes")
    parser.add_argument('--model', type=str, default='unet', help="Model type: unet")
    parser.add_argument('--dataset', type=str, default='cityscapes', help="Dataset: cityscapes")
    parser.add_argument('--data_root', type=str, default='./data/cityscapes', help="Path to Cityscapes dataset root")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 19

    val_joint_transform = SegmentationValTransform(output_size=(256,512))
    val_dataset = CityscapesDatasetWrapper(root=args.data_root, split='val', mode='fine',
                                           target_type='semantic', joint_transform=val_joint_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if args.model.lower() == 'unet':
        model = UNet(n_channels=3, n_classes=num_classes).to(device)
    else:
        raise ValueError("Unsupported model type")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    print("Loaded checkpoint:", args.checkpoint)

    avg_loss, mIoU, per_class_ious = evaluate_model(model, val_loader, device, num_classes)
    print(f"Evaluation Loss: {avg_loss:.4f}")
    print(f"Mean IoU: {mIoU:.4f}")
    for cls in range(num_classes):
        print(f"Class {cls} IoU: {per_class_ious[cls]:.4f}")

if __name__ == "__main__":
    main()
