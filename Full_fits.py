
import os
import torch
import numpy as np
from astropy.io import fits
from typing import Optional
from Unet import UNetSimple  
import matplotlib.pyplot as plt

def save_mask_as_fits(mask: np.ndarray, out_path: str, header_source: Optional[str] = None) -> None:
    hdu = fits.PrimaryHDU(mask.astype(np.float32))
    if header_source is not None:
        with fits.open(header_source) as hdul:
            hdu.header.extend(hdul[0].header, strip=True, update=True)
    hdu.writeto(out_path, overwrite=True)


#  procesar tiles de 4 FITS y reconstruir máscaras

def process_four_fits(paths, model, threshold=0.5, device='cpu'):
    data_list = []
    for path in paths:
        with fits.open(path) as hdul:
            data_list.append(hdul[0].data.astype(np.float32))

    H, W = data_list[0].shape
    masks_full = [np.zeros((H, W), dtype=np.uint8) for _ in range(4)]

    tile_size = 256
    tiles_per_row = H // tile_size
    tiles_per_col = W // tile_size

    for i in range(tiles_per_row):
        for j in range(tiles_per_col):
            tile_stack = []
            for t in range(4):
                tile = data_list[t][i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size]
                tile = (tile - tile.min()) / (tile.max() - tile.min() + 1e-8)
                tile_stack.append(tile)
            tile_tensor = torch.from_numpy(np.stack(tile_stack, axis=0)).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tile_tensor)
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).float().cpu().numpy()[0]

            for t in range(4):
                masks_full[t][i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = preds[t].astype(np.uint8)

    return masks_full, data_list  # devolvemos también los originales


def load_fits(path):
    with fits.open(path) as hdul:
        return hdul[0].data.astype(np.float32)


def compute_metrics(pred, gt, tol=1e-3):
    diff = np.abs(pred - gt)
    TP = np.sum((diff <= tol) & (gt != 0))
    TN = np.sum((diff <= tol) & (gt == 0))
    FP = np.sum((diff > tol) & (gt == 0))
    FN = np.sum((diff > tol) & (gt != 0))

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return TP, TN, FP, FN, precision, recall, f1

if __name__ == "__main__":
    output_dir = r"C:\Users\javie\Documents\GitHub\CNN_Cosmic_Rays_PS\recontruidas"
    os.makedirs(output_dir, exist_ok=True)


    recon_input_paths = [
        r"C:\Users\javie\Documents\Carrera putiversitaria\CUBESATUB\photsat_frames_cosmic_rays\photsat_frames_cosmic_rays\Img_steady_capture_35202_2200_scanning__optic_0_0.fits",
        r"C:\Users\javie\Documents\Carrera putiversitaria\CUBESATUB\photsat_frames_cosmic_rays\photsat_frames_cosmic_rays\Img_steady_capture_35203_2200_scanning__optic_0_0.fits",
        r"C:\Users\javie\Documents\Carrera putiversitaria\CUBESATUB\photsat_frames_cosmic_rays\photsat_frames_cosmic_rays\Img_steady_capture_35204_2200_scanning__optic_0_0.fits",
        r"C:\Users\javie\Documents\Carrera putiversitaria\CUBESATUB\photsat_frames_cosmic_rays\photsat_frames_cosmic_rays\Img_steady_capture_35205_2200_scanning__optic_0_0.fits",
    ]

    ground_truth_paths = [
        r"C:\Users\javie\Documents\Carrera putiversitaria\CUBESATUB\no_cosmicrays_frames\no_noisy_frames\Img_steady_capture_35202_2200_scanning__optic_0_0.fits",
        r"C:\Users\javie\Documents\Carrera putiversitaria\CUBESATUB\no_cosmicrays_frames\no_noisy_frames\Img_steady_capture_35203_2200_scanning__optic_0_0.fits",
        r"C:\Users\javie\Documents\Carrera putiversitaria\CUBESATUB\no_cosmicrays_frames\no_noisy_frames\Img_steady_capture_35204_2200_scanning__optic_0_0.fits",
        r"C:\Users\javie\Documents\Carrera putiversitaria\CUBESATUB\no_cosmicrays_frames\no_noisy_frames\Img_steady_capture_35205_2200_scanning__optic_0_0.fits",
    ]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")


    model = UNetSimple(in_channels=4, n_classes=4, base_ch=16)
    checkpoint = torch.load(
        r"C:\Users\javie\Documents\GitHub\CNN_Cosmic_Rays_PS\saved_models\model_epoch50_20251022_043759_loss0.0000.pth",
        map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()


    print("Procesando los 4 frames en tiles de 256x256...")
    masks_full, original_data_list = process_four_fits(recon_input_paths, model, threshold=0.5, device=device)


    recon_files = []
    for t in range(4):
        reconstructed = original_data_list[t] * (1 - masks_full[t])  # limpiar cosmicos
        base_name = os.path.basename(recon_input_paths[t]).replace(".fits", "_reconstructed.fits")
        out_path = os.path.join(output_dir, base_name)
        save_mask_as_fits(reconstructed, out_path, header_source=recon_input_paths[t])
        recon_files.append(out_path)
        print(f" Guardado reconstruido: {out_path}")

    total_TP = total_TN = total_FP = total_FN = 0
    for t, (recon_path, gt_path) in enumerate(zip(recon_files, ground_truth_paths)):
        # Cargar original con cosmicos y ground truth
        original = original_data_list[t]
        gt = load_fits(gt_path)

        
        true_cosmics_mask = (original - gt) > 0.0  
        pred_mask = masks_full[t] 
        TP, TN, FP, FN, precision, recall, f1 = compute_metrics(pred_mask, true_cosmics_mask)
        total_TP += TP
        total_TN += TN
        total_FP += FP
        total_FN += FN

        print(f"\nEvaluando frame {t+1} ({os.path.basename(recon_path)}):")
        print(f"TP={TP}, TN={TN}, FP={FP}, FN={FN}")
        print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for t in range(4):
        axes[t].imshow(masks_full[t], cmap='gray')
        axes[t].set_title(f"Máscara frame {t+1}")
        axes[t].axis('off')

    plt.tight_layout()
    plt.show()
