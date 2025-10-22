# evaluate_examples.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

from dataset import CosmicRaysSequenceDataset
from Unet import UNetSimple 

def evaluate_model(model_path, noisy_dir, clean_dir, num_samples=5, threshold=0.5, device="cuda"):
    # Dataset de validación
    dataset = CosmicRaysSequenceDataset(noisy_dir, clean_dir, sequence_length=4)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Cargar modelo
    model = UNetSimple(in_channels=4,  n_classes=4)  
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    samples_done = 0
    for noisy, mask in val_loader:
        noisy, mask = noisy.to(device), mask.to(device)

        # [B,T,H,W] → [B,C,H,W]
        noisy_in = noisy.unsqueeze(1)  # [B,1,T,H,W]
        noisy_in = noisy_in.permute(0,2,1,3,4).squeeze(2)  # [B,T,H,W] = [B,4,H,W]

        with torch.no_grad():
            logits = model(noisy_in)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

        # Calcular métricas por píxel
        gt = mask.cpu().numpy().astype(np.uint8)
        pr = preds.cpu().numpy().astype(np.uint8)

        TP = np.logical_and(pr == 1, gt == 1).sum()
        TN = np.logical_and(pr == 0, gt == 0).sum()
        FP = np.logical_and(pr == 1, gt == 0).sum()
        FN = np.logical_and(pr == 0, gt == 1).sum()

        precision = TP / (TP + FP + 1e-8)
        recall    = TP / (TP + FN + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        print(f"Sample {samples_done+1}")
        print(f"TP={TP}, FP={FP}, FN={FN}, TN={TN}")
        print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

        # Visualización
        fig, axs = plt.subplots(1,4, figsize=(16,4))
        axs[0].imshow(noisy[0,0].cpu(), cmap="gray")
        axs[0].set_title("Noisy (1er frame)")
        axs[1].imshow(mask[0,0].cpu(), cmap="gray")
        axs[1].set_title("GT mask")
        axs[2].imshow(probs[0,0].cpu(), cmap="inferno")
        axs[2].set_title("Pred prob")
        axs[3].imshow(preds[0,0].cpu(), cmap="gray")
        axs[3].set_title("Pred binaria")
        for ax in axs: ax.axis("off")
        plt.show()

        samples_done += 1
        if samples_done >= num_samples:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Ruta al checkpoint .pth")
    parser.add_argument("--noisy_dir", type=str, required=True)
    parser.add_argument("--clean_dir", type=str, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--thr", type=float, default=0.5)
    args = parser.parse_args()

    evaluate_model(args.model, args.noisy_dir, args.clean_dir,
                   num_samples=args.samples, threshold=args.thr)


