# train.py
import os
import torch
from torch.utils.data import DataLoader, Subset
from torch.cuda import amp
import argparse
from dataset import CosmicRaysSequenceDataset
from Unet import UNetSimple
from utils import save_checkpoint
from torch.utils.data import WeightedRandomSampler
from losses import WeightedBCEDiceLoss



def build_loaders(base_dir, batch_size=4, sequence_length=4, quick_mode=False, quick_samples=20000, num_workers=4):
    train_noisy = os.path.join(base_dir, "train", "noisy")
    train_clean = os.path.join(base_dir, "train", "clean")
    val_noisy   = os.path.join(base_dir, "val", "noisy")
    val_clean   = os.path.join(base_dir, "val", "clean")

    train_ds = CosmicRaysSequenceDataset(train_noisy, train_clean, sequence_length=sequence_length, optic_filter=0)
    val_ds   = CosmicRaysSequenceDataset(val_noisy, val_clean, sequence_length=sequence_length, optic_filter=0)

    if quick_mode:
        train_ds = Subset(train_ds, list(range(min(len(train_ds), quick_samples))))
        val_ds   = Subset(val_ds, list(range(min(len(val_ds), max(100, quick_samples//10)))))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

# build_positive_sampler(dataset) -> sampler

def build_weighted_sampler_from_dataset(dataset):
    # dataset.samples: list of sequences; compute which have any positive pixel
    weights = []
    for i in range(len(dataset)):
        noisy, mask = dataset[i]  # this loads masks we just implemented
        # mask shape [T,H,W]
        has_pos = mask.sum() > 0
        weights.append(5.0 if has_pos else 1.0)  # oversample positives 5x
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler


def train_loop(model, train_loader, val_loader, device, epochs, lr, model_dir, quick_mode=False):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = WeightedBCEDiceLoss()  # usa la combinación Focal+Dice
    scaler = amp.GradScaler()  # mixed precision

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for images, masks in train_loader:
            # images: [B, T, H, W] -> model expects channels first (B, C, H, W)
            images = images.to(device, dtype=torch.float32)
            masks  = masks.to(device, dtype=torch.float32)
            images = images.permute(0,1,2,3)              # already [B,T,H,W]
            # ensure shape [B, C, H, W]
            images = images
            # if masks are [B,T,H,W] and we want per-frame mask output channels = T
            masks = masks

            # forward/backward with AMP
            optimizer.zero_grad()
            with amp.autocast():
                logits = model(images)            # [B, n_classes, H, W]
                # If model outputs per-frame masks, make sure masks shape matches:
                # masks -> [B, T, H, W] ; criterion expects [B, n_classes, H, W]
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[{epoch}/{epochs}] Train loss: {avg_loss:.4f}")

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, dtype=torch.float32)
                masks  = masks.to(device, dtype=torch.float32)
                with amp.autocast():
                    logits = model(images)

                    loss = criterion(logits, masks)
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))
        print(f"[{epoch}/{epochs}] Val loss: {val_loss:.4f}")

        # checkpoint
        path = save_checkpoint(model, optimizer, epoch, val_loss, model_dir)
        print(f"Saved checkpoint: {path}")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\javie\Documents\GitHub\CNN_Cosmic_Rays_PS\dataset")
    parser.add_argument("--model_dir", type=str, default=r"C:\Users\javie\Documents\GitHub\CNN_Cosmic_Rays_PS\saved_models")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--quick", action="store_true", help="Quick debug run (small subset, smaller model)")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    if args.quick:
        # smaller model and fewer epochs for debug
        model = UNetSimple(in_channels=4, n_classes=4, base_ch=16)
        train_loader, val_loader = build_loaders(args.data_dir, batch_size= args.batch_size,
                                                 sequence_length=4, quick_mode=True, quick_samples=10000,
                                                 num_workers=args.num_workers)
        train_loop(model, train_loader, val_loader, device="cuda", epochs=50, lr=args.lr, model_dir=args.model_dir, quick_mode=True)
    else:
        model = UNetSimple(in_channels=4, n_classes=4, base_ch=32)
        train_loader, val_loader = build_loaders(args.data_dir, batch_size= args.batch_size,
                                                 sequence_length=4, quick_mode=False,

                                                 num_workers=args.num_workers)
        train_loop(model, train_loader, val_loader, device="cuda", epochs=args.epochs, lr=args.lr, model_dir=args.model_dir)
