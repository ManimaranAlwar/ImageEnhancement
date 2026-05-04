import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class CustomImageDataset(Dataset):
    """Paired dataset for supervised training (SIDD-style)."""
    def __init__(self, root_dir, corruption_types=None):
        self.root_dir = root_dir
        self.pairs = []
        # Each subfolder contains GT_SRGB_*.PNG and NOISY_SRGB_*.PNG
        for subdir in os.listdir(root_dir):
            subpath = os.path.join(root_dir, subdir)
            if os.path.isdir(subpath):
                gt_img = None
                noisy_img = None
                for fname in os.listdir(subpath):
                    if fname.startswith('GT_SRGB') and fname.lower().endswith('.png'):
                        gt_img = os.path.join(subpath, fname)
                    elif fname.startswith('NOISY_SRGB') and fname.lower().endswith('.png'):
                        noisy_img = os.path.join(subpath, fname)
                if gt_img and noisy_img:
                    self.pairs.append((noisy_img, gt_img))
        self.corruption_types = corruption_types

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, gt_path = self.pairs[idx]
        noisy = cv2.imread(noisy_path)
        gt = cv2.imread(gt_path)
        if noisy is None or gt is None:
            raise ValueError(f"Could not read image files: {noisy_path}, {gt_path}")
        noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        noisy = cv2.resize(noisy, (128, 128), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt, (128, 128), interpolation=cv2.INTER_AREA)
        noisy = noisy / 255.0
        gt = gt / 255.0
        if self.corruption_types:
            noisy = self.apply_corruption(noisy)
        noisy = np.transpose(noisy, (2, 0, 1)).astype(np.float32)
        gt = np.transpose(gt, (2, 0, 1)).astype(np.float32)
        return torch.tensor(noisy), torch.tensor(gt)

    def apply_corruption(self, image):
        corrupted = image.copy()
        for corruption in self.corruption_types or []:
            if corruption == 'noise':
                noise = np.random.normal(0, 0.05, corrupted.shape)
                corrupted = np.clip(corrupted + noise, 0, 1)
            elif corruption == 'blur':
                corrupted = cv2.GaussianBlur((corrupted*255).astype(np.uint8), (5,5), 0)
                corrupted = corrupted / 255.0
            elif corruption == 'low-light':
                corrupted = np.clip(corrupted * 0.5, 0, 1)
        return corrupted

class UnlabeledImageDataset(Dataset):
    """Unlabeled dataset for self-supervised training.
    Collects images recursively; if SIDD structure is present, uses NOISY_SRGB_*.PNG.
    Returns a single image tensor (C,H,W) in [0,1].
    """
    def __init__(self, root_dir, exts=(".png", ".jpg", ".jpeg")):
        self.root_dir = root_dir
        self.paths = []
        for subdir in os.listdir(root_dir):
            subpath = os.path.join(root_dir, subdir)
            if not os.path.isdir(subpath):
                continue
            files = os.listdir(subpath)
            noisy_files = [f for f in files if f.startswith('NOISY_SRGB') and f.lower().endswith('.png')]
            if noisy_files:
                for f in noisy_files:
                    self.paths.append(os.path.join(subpath, f))
            else:
                for f in files:
                    if any(f.lower().endswith(e) for e in exts):
                        self.paths.append(os.path.join(subpath, f))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read image file: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        return torch.tensor(img)
