# run_inference.py
import os
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from dataset import make_binary_mask_from_pair
from Unet import UNetSimple
from tqdm import tqdm
import re


def load_tiles_from_folder(noisy_dir, clean_dir, x_frames):
    """
    Load tiles from folders, group by tile number and sort by frame/subframe.
    Only keep tiles with exactly x_frames.
    Returns dict[tile_idx] = list of x frames arrays [T,H,W]
    """
    noisy_files = sorted(Path(noisy_dir).glob("*.png"))
    clean_files = sorted(Path(clean_dir).glob("*.png"))

    # regex pattern

    pattern = re.compile(r".*Img_steady_capture_(\d+)_.*optic_(\d)_(\d)\.fits_(\d+)_(\d)\.png$")

    groups = {}
    for nf in noisy_files:
        m = pattern.match(nf.name)
        if not m:
            continue
        frame_str, optic_str, _dummy, tile_id_str, rot_id_str = m.groups()
        frame = int(frame_str)
        tile_id = int(tile_id_str)
        rot_id = int(rot_id_str)

        if rot_id != 0:  # skip rotations
            continue

        cf = Path(clean_dir) / nf.name
        if not cf.exists():
            continue

        tile_id = int(tile_id_str)
        if tile_id not in groups:
            groups[tile_id] = []
        groups[tile_id].append((frame, nf, cf))

    # sort each tile by frame
    tile_arrays = {}
    for tid, items in groups.items():
        items.sort(key=lambda x: x[0])  # sort by frame number
        if len(items) < x_frames:
            print(f"Warning: Tile {tid} has less than {x_frames} frames, skipping")
            continue

        noisy_stack = []
        clean_stack = []
        for i in range(x_frames):
            nf, cf = items[i][1], items[i][2]
            n_img = np.array(Image.open(nf).convert("L"), dtype=np.float32) / 255.0
            c_img = np.array(Image.open(cf).convert("L"), dtype=np.float32) / 255.0


            noisy_stack.append(n_img)
            clean_stack.append(c_img)

        tile_arrays[tid] = {
            "noisy": np.stack(noisy_stack, axis=0),  # [T,H,W]
            "clean": np.stack(clean_stack, axis=0)   # [T,H,W]
        }

    return tile_arrays

def preprocess_tiles(tile_arrays, k_sigma=6.0, min_abs=0.002, dilate_pixels=0, max_area=16):
    """
    For each tile compute ground truth mask from noisy + clean pair.
    Returns dict[tile_idx] = dict("noisy": ..., "mask": ...)
    """
    processed = {}
    for tid, data in tile_arrays.items():
        noisy, clean = data["noisy"], data["clean"]
        masks = []
        for n_img, c_img in zip(noisy, clean):
            mask, thr = make_binary_mask_from_pair(n_img, c_img, k_sigma=k_sigma, min_abs=min_abs,
                                                   dilate_pixels=dilate_pixels, max_area=max_area)
            masks.append(mask.astype(np.float32))
        processed[tid] = {
            "noisy": noisy.astype(np.float32),
            "mask": np.stack(masks, axis=0)  # [T,H,W]
        }
    return processed

def build_model_inputs(processed_tiles):
    """
    Convert tiles to torch tensors [T,H,W] -> [1,T,H,W] for model batch processing
    """
    inputs = {}
    for tid, data in processed_tiles.items():
        inputs[tid] = torch.from_numpy(data["noisy"]).float()  # [T,H,W]
    return inputs

def predict_masks(model, model_inputs, device="cuda", threshold=0.001):
    """
    Run model tile by tile, apply threshold to get binary mask
    Returns dict[tile_idx] = [T,H,W] numpy arrays
    """
    model.eval()
    pred_masks = {}
    with torch.no_grad():
        for tid, tile in model_inputs.items():
            x = tile.unsqueeze(0).to(device)  # [1,T,H,W]
            logits = model(x)                  # [1,T,H,W] or [1,n_classes,H,W]
            probs = torch.sigmoid(logits)


            pred = (probs > threshold).float()
            pred_masks[tid] = pred.squeeze(0).cpu().numpy()  # [T,H,W]
    return pred_masks

def evaluate_model_masks(predicted_masks, ground_truth_masks):
    """
    Compare predicted binary masks with ground truth
    """
    TP = TN = FP = FN = 0
    for tid in predicted_masks.keys():
        pred = predicted_masks[tid]
        gt   = ground_truth_masks[tid]["mask"]
        for p, g in zip(pred, gt):
            TP += np.sum((p==1) & (g==1))
            TN += np.sum((p==0) & (g==0))
            FP += np.sum((p==1) & (g==0))
            FN += np.sum((p==0) & (g==1))
    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * (precision*recall) / (precision + recall + 1e-8)
    stats = {"TP":TP, "TN":TN, "FP":FP, "FN":FN, "Precision":precision, "Recall":recall, "F1":f1}
    return stats

def reconstruct_full_image(tile_dict):
    """
    Reconstruct 2048x2048 full image from 64 tiles in raster order
    """
    T,H,W = list(tile_dict.values())[0].shape
    full_images = []
    rows, cols = 8,8
    for t in range(T):
        img = np.zeros((rows*H, cols*W), dtype=np.float32)
        for idx in range(64):
            r, c = divmod(idx, cols)
            if idx in tile_dict:
                img[r*H:(r+1)*H, c*W:(c+1)*W] = tile_dict[idx][t]
        full_images.append(img)
    return np.stack(full_images, axis=0)

# -----------------------------
# Main script
# -----------------------------

if __name__ == "__main__":
    noisy_dir = r"C:\Users\javie\Documents\GitHub\CNN_Cosmic_Rays_PS\dataset\test\noisy"
    clean_dir = r"C:\Users\javie\Documents\GitHub\CNN_Cosmic_Rays_PS\dataset\test\clean"
    x_frames = 4
    threshold = 0.5
    device = "cuda"

    # Load and preprocess tiles
    tiles = load_tiles_from_folder(noisy_dir, clean_dir, x_frames)
    processed = preprocess_tiles(tiles)

    # Build model inputs
    model_inputs = build_model_inputs(processed)

    

    # Load model
    model = UNetSimple(in_channels=x_frames, n_classes=x_frames, base_ch=16)
    checkpoint = torch.load(r"C:\Users\javie\Documents\GitHub\CNN_Cosmic_Rays_PS\saved_models\model_epoch50_20251022_043759_loss0.0000.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from epoch {checkpoint['epoch']} with val loss {checkpoint['loss']:.4f}")

    model.to(device)

# after model.to(device)
for tid, t in model_inputs.items():
    print("tile", tid, "tensor shape", t.shape)  # expect [T,H,W]
    break

# print model forward shape on one batch
x = next(iter(model_inputs.values())).unsqueeze(0).to(device)  # [1,T,H,W]
with torch.no_grad():
    logits = model(x)
    print("logits shape:", logits.shape)  # expect [1, n_classes, H, W] where n_classes = T
    for tile_idx, stack in tiles.items():
        noisy = stack['noisy']  # [T,H,W]
        clean = stack['clean']  # [T,H,W]
        print(f"Tile {tile_idx}: noisy shape {noisy.shape}, clean shape {clean.shape}")
        break  # just first tile for debug


    # Predict masks
    pred_masks = predict_masks(model, model_inputs, device=device, threshold=threshold)

    # Check predicted probability distribution
    all_probs = np.concatenate([p.flatten() for p in pred_masks.values()])
    print("Predicted probability stats -> Min:", all_probs.min(), "Mean:", all_probs.mean(), "Max:", all_probs.max())

    # Evaluate
    stats = evaluate_model_masks(pred_masks, processed)
    print("Evaluation stats:", stats)

    #  reconstruct full 2048x2048 mask
    full_mask = reconstruct_full_image(pred_masks)
    print("Full reconstructed mask shape:", full_mask.shape)


