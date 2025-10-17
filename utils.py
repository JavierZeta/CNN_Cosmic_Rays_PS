# utils.py he quitado la exception de dilate y he puesto 0 por defecto, he quitado exception filter by area 
import os
import torch
from datetime import datetime
import numpy as np

def save_checkpoint(model, optimizer, epoch, loss, model_dir, prefix="model"):
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(model_dir, f"{prefix}_epoch{epoch}_{timestamp}_loss{loss:.4f}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    return path

def robust_mad_sigma(img):
    med = np.median(img)
    mad = np.median(np.abs(img - med))
    return 1.4826 * mad

def filter_by_area(mask, max_area=16):
    """
    Keep only connected components with area <= max_area.
    Returns filtered boolean mask.
    """

    from scipy.ndimage import label
    labeled, n = label(mask) ## n storage the number of connected components
    out = np.zeros_like(mask, dtype=bool)
    for lab in range(1, n + 1):
        comp = (labeled == lab)
        if comp.sum() <= max_area:
            out |= comp
    return out

def make_binary_mask_from_pair(noisy, clean, k_sigma=6.0, min_abs=0.01, dilate_pixels=1, max_area=16):
    """
    Create binary mask of cosmic rays from a noisy/clean image pair.
    - Uses diff = max(noisy - clean, 0)
    - thr = max(k_sigma * sigma_clean, min_abs)
    - small dilation to cover neighbors if needed
    - filters out large connected components (> max_area)
    Returns: mask (bool), thr_used (float)
    """
    diff = noisy - clean
    diff = np.clip(diff, 0.0, None)

    sigma = robust_mad_sigma(clean)
    if sigma <= 1e-8:
        thr = float(min_abs)
    else:
        thr = float(max(k_sigma * sigma, min_abs))

    mask = diff > thr

    # optional small dilation
    if dilate_pixels and mask.any(): ### In case that we think that the cosmic ray affects neighbor pixels
            from scipy.ndimage import binary_dilation
            mask = binary_dilation(mask, iterations=dilate_pixels)


    # filter out large objects (likely stars)
    if max_area is not None and mask.any():
        mask = filter_by_area(mask, max_area=max_area)

    return mask.astype(np.uint8), thr