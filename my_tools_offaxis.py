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


class DirectInterference_torch:
    """
    Direct Off-Axis Interference / Intensity Model at Image Plane.
    Simulates interference of object field U = exp(i * pi * ph) 
    with reference beams R_c = exp(i * (kx_c * x + ky_c * y)).
    """
    def __init__(self, comp_field, params):
        self.comp_field = comp_field  # [H, W] complex field
        self.n_pixel = params['patch_size']
        self.pixel_size = params['pixel_size']

        L = self.n_pixel * self.pixel_size
        x_lin = np.linspace(-L / 2, L / 2, self.n_pixel, endpoint=False, dtype=np.float32)
        y_lin = np.linspace(-L / 2, L / 2, self.n_pixel, endpoint=False, dtype=np.float32)
        xx, yy = np.meshgrid(x_lin, y_lin)
        self.x = torch.from_numpy(xx).to(device)
        self.y = torch.from_numpy(yy).to(device)

    def __call__(self, angle_param):
        kx, ky = angle_param
        ref_beam = torch.exp(1j * (kx * self.x + ky * self.y))
        intensity = torch.abs(self.comp_field + ref_beam) ** 2
        return intensity


def batch_forward_direct(batch_phase, angles_batch, params):
    """
    batch_phase: [N, 1, H, W] predicted phase map
    angles_batch: list of angle parameters [(kx1, ky1), (kx2, ky2)]
    Returns: [N, C=2, H, W] simulated intensity holograms
    """
    N, C_out, H, W = batch_phase.shape
    simulated_batch = []

    for n in range(N):
        ph = batch_phase[n, 0, ...]
        comp_field = torch.exp(1j * params['ph'] * np.pi * ph)
        forward_model = DirectInterference_torch(comp_field, params)

        channel_imgs = []
        for c in range(len(angles_batch[n])):
            kx, ky = angles_batch[n][c]
            intensity = forward_model((kx, ky))
            channel_imgs.append(intensity)

        simulated_batch.append(torch.stack(channel_imgs, dim=0))

    return torch.stack(simulated_batch, dim=0)


class RealHoloTrainDataset(torch.utils.data.Dataset):
    """
    Dataset to train GedankenNet directly on real experimental BMP holograms from data_raw.
    Supports file pairing like sam_(1).bmp and sam_(2).bmp.
    Includes a robust fallback generator so DataLoader NEVER fails with num_samples=0 even if data_raw is empty.
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
            # Fallback dynamic synthetic hologram generator if no files uploaded yet
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
