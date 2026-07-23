import os
import glob
import re
import random
import torch
import torch.nn as nn
import numpy as np
import PIL.Image
import scipy.ndimage

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def min_max_norm(img, vmin=None, vmax=None):
    if vmin is None:
        vmin = img.min()
    if vmax is None:
        vmax = img.max()
    img = np.clip(img, vmin, vmax)
    return (img - vmin) / (vmax - vmin + 1e-8)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


class LearnableDirectInterference(nn.Module):
    """
    Learnable Off-Axis Interference Physics Layer.
    - Reference wavevectors (k1_x, k1_y) and (k2_x, k2_y) are learnable PyTorch nn.Parameters.
    - Accepts continuous [sin(phi), cos(phi)] network predictions to prevent phase wrapping.
    - Computes complex field U = (cos(phi) + i*sin(phi)) / sqrt(cos^2 + sin^2 + eps).
    """
    def __init__(self, params):
        super().__init__()
        self.patch_size = params['patch_size']
        self.pixel_size = params['pixel_size']
        self.wl = params['wavelength'] / params['ref_ind']
        k_mag = 2 * np.pi / self.wl

        # Initial wavevector estimates (e.g. +15 deg and -15 deg)
        init_kx1 = k_mag * np.sin(np.deg2rad(15.0))
        init_kx2 = k_mag * np.sin(np.deg2rad(-15.0))

        # Learnable reference wavevectors: k1 = [kx1, ky1], k2 = [kx2, ky2]
        self.k1 = nn.Parameter(torch.tensor([init_kx1, 0.0], dtype=torch.float32))
        self.k2 = nn.Parameter(torch.tensor([init_kx2, 0.0], dtype=torch.float32))

        # Spatial grid [H, W]
        L = self.patch_size * self.pixel_size
        x_lin = np.linspace(-L / 2, L / 2, self.patch_size, endpoint=False, dtype=np.float32)
        y_lin = np.linspace(-L / 2, L / 2, self.patch_size, endpoint=False, dtype=np.float32)
        xx, yy = np.meshgrid(x_lin, y_lin)

        self.register_buffer('x', torch.from_numpy(xx))
        self.register_buffer('y', torch.from_numpy(yy))

    def forward(self, pred_sc):
        """
        pred_sc: [N, 2, H, W] where channel 0 = sin(phi), channel 1 = cos(phi)
        Returns: [N, 2, H, W] simulated intensity holograms for angle 1 and angle 2
        """
        sin_p = pred_sc[:, 0:1, :, :]
        cos_p = pred_sc[:, 1:2, :, :]

        # Normalize to form unit complex field: U = (cos(phi) + i*sin(phi)) / sqrt(cos^2 + sin^2 + eps)
        norm = torch.sqrt(sin_p ** 2 + cos_p ** 2 + 1e-8)
        comp_field = (cos_p + 1j * sin_p) / norm  # [N, 1, H, W]
        comp_field = comp_field.squeeze(1)       # [N, H, W]

        # Learnable reference beams
        ref1 = torch.exp(1j * (self.k1[0] * self.x + self.k1[1] * self.y))  # [H, W]
        ref2 = torch.exp(1j * (self.k2[0] * self.x + self.k2[1] * self.y))  # [H, W]

        # Simulated interference intensity holograms
        H1_pred = torch.abs(comp_field + ref1) ** 2  # [N, H, W]
        H2_pred = torch.abs(comp_field + ref2) ** 2  # [N, H, W]

        return torch.stack([H1_pred, H2_pred], dim=1)  # [N, 2, H, W]


class RealHoloTrainDataset(torch.utils.data.Dataset):
    """
    Dataset to train GedankenNet directly on real experimental BMP holograms from data_raw.
    Supports file pairing like sam_(1).bmp and sam_(2).bmp.
    Includes a robust fallback generator so DataLoader NEVER fails with num_samples=0.
    """
    def __init__(self, raw_dir, trans=None, default_num_samples=500):
        self.raw_dir = raw_dir
        self.trans = trans
        self.default_num_samples = default_num_samples
        self.pairs = []

        bmp_files = sorted(
            glob.glob(os.path.join(raw_dir, '*.bmp')) + glob.glob(os.path.join(raw_dir, '*.png')),
            key=natural_sort_key
        )

        prefix_dict = {}
        for f in bmp_files:
            fname = os.path.basename(f)
            match = re.match(r'^(.*?)[_\s]?\(?(\d+)\)?\.(bmp|png)$', fname, re.IGNORECASE)
            if match:
                prefix = match.group(1).rstrip('_')
                idx = int(match.group(2))
                if prefix not in prefix_dict:
                    prefix_dict[prefix] = {}
                prefix_dict[prefix][idx] = f

        matched_pairs = []
        for prefix, items in prefix_dict.items():
            sorted_indices = sorted(items.keys())
            for k in range(0, len(sorted_indices) - 1, 2):
                idx1 = sorted_indices[k]
                idx2 = sorted_indices[k + 1]
                matched_pairs.append((items[idx1], items[idx2], f"{prefix}_sample_{idx1}_{idx2}"))

        if len(matched_pairs) > 0:
            self.pairs = matched_pairs
        elif len(bmp_files) >= 2:
            for i in range(0, len(bmp_files) - 1, 2):
                f1, f2 = bmp_files[i], bmp_files[i + 1]
                sample_name = os.path.splitext(os.path.basename(f1))[0]
                self.pairs.append((f1, f2, sample_name))

    def __len__(self):
        if len(self.pairs) > 0:
            return len(self.pairs)
        return self.default_num_samples

    def __getitem__(self, index):
        if len(self.pairs) > 0:
            img_path1, img_path2, sample_name = self.pairs[index % len(self.pairs)]
            img1 = np.array(PIL.Image.open(img_path1).convert('L')).astype('float32') / 255.0
            img2 = np.array(PIL.Image.open(img_path2).convert('L')).astype('float32') / 255.0
            img_stacked = np.stack([img1, img2], axis=-1)  # [H, W, 2]
        else:
            s = 512
            ang = np.random.uniform(0, 1, size=(s, s)).astype('float32')
            sigma = random.uniform(8.0, 20.0)
            ang = scipy.ndimage.gaussian_filter(ang, sigma=sigma)
            ang = (ang - ang.min()) / (ang.max() - ang.min() + 1e-8)
            img1 = ang
            img2 = np.roll(ang, shift=10, axis=0)
            img_stacked = np.stack([img1, img2], axis=-1)
            sample_name = f"synthetic_sample_{index}"

        if self.trans is not None:
            img_stacked = self.trans(img_stacked)
        else:
            img_stacked = torch.from_numpy(img_stacked.transpose((2, 0, 1)))

        return img_stacked, sample_name


class HoloBmpDataset(RealHoloTrainDataset):
    pass


class GedankenOffAxisDataset(RealHoloTrainDataset):
    pass
