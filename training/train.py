import os
import sys
import time
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from utils.dataset import CityscapesDatasetWrapper
from utils.transforms import SegmentationTrainTransform, SegmentationValTransform
from utils.losses import HybridLoss
from utils.metrics import compute_mIoU
from utils.map_trainId_imgs import convert_id_to_train_id
from utils.logger import Logger
from models.unet import UNet
from models.deeplabv3 import DeepLabV3Plus
from models.deeplabv3_attention import DeepLabV3PlusWithAttention
from models.unet_attention import UNetWithAttention
from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs, device, num_classes, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    logger = Logger(log_dir)
    best_mIoU = 0.0
    global_step = 0

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0

        for i, (images, masks) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}")):
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
        logger.info(f"Training Loss: {epoch_loss:.4f}")
        writer.add_scalar("Train/Epoch_Loss", epoch_loss, epoch)

        model.eval()
        valid_loss = 0.0
        total_ious = np.zeros(num_classes)
        count_ious = np.zeros(num_classes)

        with torch.no_grad():
            for images, masks in tqdm(valid_loader, desc=f"Val Epoch {epoch+1}"):
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                # print("Mask dtype:", masks.dtype)  # Should be torch.long
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
            logger.info(f"Validation Loss: {valid_loss:.4f} - mIoU: {mIoU:.4f}")
            writer.add_scalar("Valid/mIoU", mIoU, epoch)
            for cls in range(num_classes):
                writer.add_scalar(f"Valid/IoU_Class_{cls}", avg_ious[cls], epoch)

        scheduler.step()

        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
        writer.add_scalar("Epoch/Time", epoch_time, epoch)

        if mIoU > best_mIoU:
            best_mIoU = mIoU
            best_model_path = os.path.join(log_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            logger.info("Saved Best Model")

    writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train segmentation model on Cityscapes")
    parser.add_argument('--model', type=str, default='unet', help="Model type: unet or deeplabv3")
    parser.add_argument('--dataset', type=str, default='cityscapes', help="Dataset: cityscapes")
    parser.add_argument('--data_root', type=str, default='./data/cityscapes', help="Path to Cityscapes dataset root")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--log_dir', type=str, default='./runs/segmentation_experiment', help="Directory to save logs and best model")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to a checkpoint to resume training")

    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = 19

    set_seed(42)
    os.makedirs(args.log_dir, exist_ok=True)

    train_transform = SegmentationTrainTransform(base_size=(512, 256), scale_range=(1.0, 1.5))
    val_transform = SegmentationValTransform(resize_to=(256, 512))

    train_dataset = CityscapesDatasetWrapper(root=args.data_root, split='train', mode='fine', target_type='semantic', joint_transform=train_transform)
    valid_dataset = CityscapesDatasetWrapper(root=args.data_root, split='val', mode='fine', target_type='semantic', joint_transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if args.model.lower() == 'unet':
        model = UNet(n_channels=3, n_classes=num_classes).to(device)
    elif args.model.lower() == 'unet_attention':
        model = UNetWithAttention(n_channels=3, n_classes=num_classes).to(device)
    elif args.model.lower() == 'deeplabv3':
        model = DeepLabV3Plus(num_classes=num_classes, backbone='resnet50', pretrained_backbone=True).to(device)
    elif args.model.lower() == 'deeplabv3_attention':
        model = DeepLabV3PlusWithAttention(num_classes=num_classes, backbone='mobilenetv2', pretrained_backbone=True).to(device)
    else:
        raise ValueError("Unsupported model type: {}".format(args.model))

    if args.checkpoint is not None:
        print("Resuming training from checkpoint:", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint)

    # criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    criterion = HybridLoss(weight_focal=0.7, weight_dice=0.3, alpha=0.5, gamma=1.5, smooth=0.1, ignore_index=255, num_classes=num_classes)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)      
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-5)

    train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler,
                args.num_epochs, device, num_classes, args.log_dir)


if __name__ == "__main__":
    main()