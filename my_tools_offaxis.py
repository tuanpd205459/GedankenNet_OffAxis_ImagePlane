import os
import glob
import re
import torch
import torch.nn as nn
import numpy as np
import PIL.Image

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
    NO Wave Propagation (no FFT/Fresnel/Angular Spectrum).
    NO Lens Pupil Filtering.

    Simulates the interference of object field U = exp(i * pi * ph) 
    with reference beams R_c = A_r * exp(i * (kx_c * x + ky_c * y)).
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
        """
        angle_param: (kx, ky) wavevector of the off-axis reference beam/angle.
        """
        kx, ky = angle_param

        # Off-axis reference beam at angle (kx, ky)
        ref_beam = torch.exp(1j * (kx * self.x + ky * self.y))

        # Direct interference intensity at image plane: I = |U_obj + R|^2
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


class GedankenOffAxisDataset(torch.utils.data.Dataset):
    """
    Synthetic dataset generator for GedankenNet direct off-axis training.
    """
    def __init__(self, file_paths, angles_list, trans, params):
        self.file_paths = file_paths
        self.angles_list = angles_list
        self.trans = trans
        self.params = params

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        ang_img = np.array(PIL.Image.open(self.file_paths[index])).astype('float32') / 255.0
        ang = self.trans(ang_img[:, :, 0]).numpy().squeeze()

        comp_field = np.exp(1j * self.params['ph'] * np.pi * ang)
        comp_field_tensor = torch.from_numpy(comp_field).to(device)
        forward_model = DirectInterference_torch(comp_field_tensor, self.params)

        imgs = []
        for kx, ky in self.angles_list:
            intensity = forward_model((kx, ky)).cpu().numpy()
            imgs.append(intensity)

        inp = np.stack(imgs, axis=0).astype('float32')  # [2, H, W]
        gt_phase = ang[np.newaxis, ...].astype('float32')  # [1, H, W]
        angles_tensor = torch.tensor(self.angles_list, dtype=torch.float32)

        return torch.Tensor(inp), torch.Tensor(gt_phase), angles_tensor


class HoloBmpDataset(torch.utils.data.Dataset):
    """
    Loads real experimental BMP holograms from data_raw folder.
    Supports filenames like:
      - sam_(1).bmp, sam_(2).bmp  -> Form sample 'sam' with angle 1 & 2
      - sam_a.bmp, sam_b.bmp
      - Sequential pairs sorted naturally
    """
    def __init__(self, raw_dir, trans=None):
        self.raw_dir = raw_dir
        self.trans = trans
        self.pairs = []

        bmp_files = sorted(
            glob.glob(os.path.join(raw_dir, '*.bmp')) + glob.glob(os.path.join(raw_dir, '*.png')),
            key=natural_sort_key
        )

        # Smart pairing logic:
        # Group by base prefix if files follow pattern like `prefix_(1).bmp` & `prefix_(2).bmp`
        prefix_dict = {}
        for f in bmp_files:
            fname = os.path.basename(f)
            # Match pattern like name_(1).bmp or name_1.bmp or name(1).bmp
            match = re.match(r'^(.*?)[_\s]?\(?(\d+)\)?\.(bmp|png)$', fname, re.IGNORECASE)
            if match:
                prefix = match.group(1).rstrip('_')
                idx = int(match.group(2))
                if prefix not in prefix_dict:
                    prefix_dict[prefix] = {}
                prefix_dict[prefix][idx] = f

        # Try building pairs from matched pattern
        matched_pairs = []
        for prefix, items in prefix_dict.items():
            sorted_indices = sorted(items.keys())
            for k in range(0, len(sorted_indices) - 1, 2):
                idx1 = sorted_indices[k]
                idx2 = sorted_indices[k + 1]
                matched_pairs.append((items[idx1], items[idx2], f"{prefix}_sample_{idx1}_{idx2}"))

        if len(matched_pairs) > 0:
            self.pairs = matched_pairs
        else:
            # Fallback to sequential pairing
            for i in range(0, len(bmp_files) - 1, 2):
                f1, f2 = bmp_files[i], bmp_files[i + 1]
                sample_name = os.path.splitext(os.path.basename(f1))[0]
                self.pairs.append((f1, f2, sample_name))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        img_path1, img_path2, sample_name = self.pairs[index]

        img1 = np.array(PIL.Image.open(img_path1).convert('L')).astype('float32') / 255.0
        img2 = np.array(PIL.Image.open(img_path2).convert('L')).astype('float32') / 255.0

        # Stack into 2-channel array
        img_stacked = np.stack([img1, img2], axis=-1)

        if self.trans is not None:
            img_stacked = self.trans(img_stacked)  # [2, H, W]
        else:
            img_stacked = torch.from_numpy(img_stacked.transpose((2, 0, 1)))

        return img_stacked, sample_name
