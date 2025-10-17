# dataset.py
"""
CosmicRaysSequenceDataset
-------------------------
Dataset that returns sequences of consecutive tiles (frames) and a binary mask
per frame that indicates cosmic-ray pixels (1) vs background (0).

The dataset expects PNG files named using the pattern produced by your preprocessing:
  ...Img_steady_capture_<FRAME>_...optic_<OPTIC>_<X>.fits_<TILEID>_<ROT>.png

It groups tiles by (tile_id, rot_id, optic) and only builds sliding windows of
`sequence_length` when the frame numbers are strictly consecutive.

Returned sample:
    noisy_tensor: torch.FloatTensor, shape [T, H, W]  (values 0..1)
    mask_tensor:  torch.FloatTensor, shape [T, H, W]  (binary 0.0/1.0)

Parameters (constructor):
- noisy_dir, clean_dir : paths to directories (noisy and corresponding clean images)
- sequence_length : number of consecutive frames (default 4)
- optic_filter : only use files with this optic (default 0 => visible)
- k_sigma : multiplier for MAD sigma -> threshold = max(k_sigma * sigma, min_abs)
- min_abs : minimum absolute threshold in normalized units (0..1)
- dilate_pixels : number of dilation iterations to expand mask (small int)
- max_area : remove connected components with area > max_area (to remove stars)
- save_masks : if True, save generated mask PNGs to `save_dir` for inspection
- save_dir : directory where masks are saved if save_masks=True
- transform : optional callable applied to (noisy, mask) numpy arrays before
              converting to tensors (useful for normalization or torch transforms)
"""

import re
from pathlib import Path
from collections import defaultdict
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import make_binary_mask_from_pair


class CosmicRaysSequenceDataset(Dataset):
    def __init__(
        self,
        noisy_dir,
        clean_dir,
        sequence_length: int = 4,
        optic_filter: int = 0,
        transform=None,
        k_sigma: float = 6.0,
        min_abs: float = 0.002,
        dilate_pixels: int = 0,
        max_area: int = 16,
        save_masks: bool = False,
        save_dir: str = None,
        verbose: bool = False,
    ):
        self.noisy_dir = Path(noisy_dir)
        self.clean_dir = Path(clean_dir)
        self.sequence_length = int(sequence_length)
        self.optic_filter = int(optic_filter)
        self.transform = transform

        # mask generation parameters
        self.k_sigma = float(k_sigma)
        self.min_abs = float(min_abs)
        self.dilate_pixels = int(dilate_pixels)
        self.max_area = None if max_area is None else int(max_area)

        # saving masks for inspection
        self.save_masks = bool(save_masks)
        if save_dir is None:
            self.save_dir = Path("./generated_masks")
        else:
            self.save_dir = Path(save_dir)
        if self.save_masks:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.verbose = bool(verbose)

        # list PNGs (assumes dataset already preprocessed to PNG)
        self.noisy_files = sorted(self.noisy_dir.glob("*.png"))
        self.clean_files = sorted(self.clean_dir.glob("*.png"))

        if self.verbose:
            print(f"[Dataset] noisy files found: {len(self.noisy_files)}")
            print(f"[Dataset] clean files found: {len(self.clean_files)}")

        # regex pattern to parse filenames
        # example: Img_steady_capture_35212_2200_scanning__optic_0_0.fits_22_1.png
        pattern = re.compile(r".*Img_steady_capture_(\d+)_.*optic_(\d)_(\d)\.fits_(\d+)_(\d+)\.png$")

        # group by (tile_id, rot_id, optic)
        groups = defaultdict(list)
        matched = 0
        for nf in self.noisy_files:
            cf = self.clean_dir / nf.name# it searches for the same name in clean dir
            if not cf.exists():
                # no pair found: skip
                continue
            m = pattern.match(nf.name)
            if not m:
                # name doesn't match expected pattern: skip
                continue

            frame_str, optic_str, _dummy, tile_id_str, rot_id_str = m.groups()
            frame = int(frame_str)
            optic = int(optic_str)
            tile_id = int(tile_id_str)
            rot_id = int(rot_id_str)

            if optic != self.optic_filter:
                continue

            groups[(tile_id, rot_id)].append((frame, nf, cf))
            matched += 1

        if self.verbose:
            print(f"[Dataset] matched (paired) noisy/clean files (optic={self.optic_filter}): {matched}")
            print(f"[Dataset] groups created: {len(groups)}")

        # sort each group by frame and build sliding windows only when frames are consecutive
        self.samples = []
        for key, items in groups.items():
            items.sort(key=lambda x: x[0])  # sort by frame number
            frames = [it[0] for it in items]
            paths = [(it[1], it[2]) for it in items]
            L = len(frames)
            if L < self.sequence_length:
                continue
            for i in range(L - self.sequence_length + 1):
                window_frames = frames[i : i + self.sequence_length]
                diffs = np.diff(window_frames)### mira que la diferencia sea 1 entre frames
                if np.all(diffs == 1):
                    window_paths = paths[i : i + self.sequence_length]
                    self.samples.append(window_paths)

        if self.verbose:
            print(f"[Dataset] total sequences found (consecutive): {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Return:
            noisy_tensor: torch.FloatTensor [T, H, W]  (0..1)
            mask_tensor:  torch.FloatTensor [T, H, W]  (0.0/1.0)
        """
        seq = self.samples[idx]  # list length T of (noisy_path, clean_path)
        noisy_stack = []
        mask_stack = []

        for nf, cf in seq:
            # load as single-channel float arrays normalized to 0..1
            n_img = Image.open(nf).convert("L")
            c_img = Image.open(cf).convert("L")
            n_arr = np.array(n_img, dtype=np.float32) / 255.0
            c_arr = np.array(c_img, dtype=np.float32) / 255.0

            # call your robust mask generator (from utils)
            mask, thr_used = make_binary_mask_from_pair(
                noisy=n_arr,
                clean=c_arr,
                k_sigma=self.k_sigma,
                min_abs=self.min_abs,
                dilate_pixels=self.dilate_pixels,
                max_area=self.max_area,
            )

            # optionally save mask PNG for inspection
            if self.save_masks:
                out_name = f"{Path(nf).stem}_mask.png"
                out_path = self.save_dir / out_name
                try:
                    # mask is 0/1 uint8
                    Image.fromarray((mask * 255).astype(np.uint8)).save(out_path)
                except Exception as e:
                    if self.verbose:
                        print(f"[Dataset] failed to save mask {out_path}: {e}")

            noisy_stack.append(n_arr)
            mask_stack.append(mask.astype(np.float32))

        # stack: shape [T, H, W]
        noisy_np = np.stack(noisy_stack, axis=0).astype(np.float32)
        mask_np = np.stack(mask_stack, axis=0).astype(np.float32)

        # apply optional transform (transform should accept noisy, mask numpy arrays and return transformed)
        if self.transform is not None:
            transformed = self.transform(noisy_np, mask_np)
            # transform expected to return (noisy_np, mask_np) in numpy form or torch tensors
            if isinstance(transformed, tuple) and len(transformed) == 2:
                noisy_np, mask_np = transformed
            else:
                # if transform returned a tensor for noisy only, try to keep mask as-is
                noisy_np = transformed

        # convert to torch tensors
        noisy_tensor = torch.from_numpy(noisy_np)  # shape [T, H, W]
        mask_tensor = torch.from_numpy(mask_np)    # shape [T, H, W]

        # ensure float32
        noisy_tensor = noisy_tensor.float()
        mask_tensor = mask_tensor.float()

        return noisy_tensor, mask_tensor
