#!/usr/bin/env python
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import CityscapesDatasetWrapper
from utils.transforms import SegmentationTrainTransform, SegmentationValTransform
from utils.losses import HybridLoss
from utils.metrics import compute_mIoU
from models.unet import UNet
from models.deeplabv3 import DeepLabV3Plus

# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs, device, num_classes, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    best_mIoU = 0.0
    global_step = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            global_step += 1

        epoch_loss = running_loss / len(train_loader)
        print(f"Training Loss: {epoch_loss:.4f}")
        writer.add_scalar("Train/Epoch_Loss", epoch_loss, epoch)

        model.eval()
        valid_loss = 0.0
        total_ious = np.zeros(num_classes)
        count_ious = np.zeros(num_classes)
        with torch.no_grad():
            for images, masks in valid_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                valid_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                per_class_ious, _ = compute_mIoU(preds.cpu().numpy(), masks.cpu().numpy(), num_classes)
                for cls in range(num_classes):
                    if not np.isnan(per_class_ious[cls]):
                        total_ious[cls] += per_class_ious[cls]
                        count_ious[cls] += 1

            valid_loss /= len(valid_loader)
            writer.add_scalar("Valid/Loss", valid_loss, epoch)
            avg_ious = [total_ious[i] / count_ious[i] if count_ious[i] > 0 else 0 for i in range(num_classes)]
            mIoU = np.mean(avg_ious)
            print(f"Validation Loss: {valid_loss:.4f} - mIoU: {mIoU:.4f}")
            writer.add_scalar("Valid/mIoU", mIoU, epoch)
            for cls in range(num_classes):
                writer.add_scalar(f"Valid/IoU_Class_{cls}", avg_ious[cls], epoch)

        scheduler.step()
        # Save checkpoint for each epoch
        checkpoint_path = os.path.join(log_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        if mIoU > best_mIoU:
            best_mIoU = mIoU
            best_model_path = os.path.join(log_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print("Saved Best Model")

    writer.close()

def main():
    parser = argparse.ArgumentParser(description="Train U-Net on Cityscapes")
    parser.add_argument('--model', type=str, default='unet', help="Model type: unet")
    parser.add_argument('--dataset', type=str, default='cityscapes', help="Dataset: cityscapes")
    parser.add_argument('--data_root', type=str, default='./data/cityscapes', help="Path to Cityscapes dataset root")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--log_dir', type=str, default='./runs/unet_experiment', help="Directory to save logs and checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 19  # Cityscapes has 19 classes

    os.makedirs(args.log_dir, exist_ok=True)

    train_joint_transform = SegmentationTrainTransform(output_size=(256,512))
    val_joint_transform = SegmentationValTransform(output_size=(256,512))

    train_dataset = CityscapesDatasetWrapper(root=args.data_root, split='train', mode='fine',
                                               target_type='semantic', joint_transform=train_joint_transform)
    valid_dataset = CityscapesDatasetWrapper(root=args.data_root, split='val', mode='fine',
                                               target_type='semantic', joint_transform=val_joint_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if args.model.lower() == 'unet':
        model = UNet(n_channels=3, n_classes=num_classes).to(device)
    else:
        raise ValueError("Unsupported model type")

    criterion = HybridLoss(weight_focal=0.5, weight_dice=0.5, alpha=0.25, gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler,
                args.num_epochs, device, num_classes, args.log_dir)

if __name__ == "__main__":
    main()
