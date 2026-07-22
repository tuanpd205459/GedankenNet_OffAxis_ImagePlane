###############################################################
#  GedankenNet-Phase: Self-Supervised Training on Real BMP Data
#  Directly trains FNO2d using Real Holograms in /data_raw
###############################################################

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np
import torch.nn as nn
import glob
from timeit import default_timer
from torch.utils.tensorboard import SummaryWriter

from utilities import count_params, device
from networks.fno import FNO2d
from my_tools_offaxis import batch_forward_direct, RealHoloTrainDataset, GedankenOffAxisDataset
import np_transforms
from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)

# ==============================================================
# CONFIGURATIONS
# ==============================================================
RAW_DIR = 'data_raw'
TRAIN_PATH = 'data/train'

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(TRAIN_PATH, exist_ok=True)

S = 512  # Patch crop size (512x512)
modes = 256
width = 4

batch_size = 1
epochs = 1000
batch_per_ep = 100
learning_rate = 0.0001

params = {
    'wavelength': 0.530,    # um
    'pixel_size': 0.3733,   # um
    'patch_size': S,
    'ref_ind': 1.00,
    'ph': 1.0
}

# 2 Off-Axis Reference/Illumination Wavevectors (kx, ky) in rad/um
wavelength = params['wavelength'] / params['ref_ind']
k_mag = 2 * np.pi / wavelength
theta1, theta2 = np.deg2rad(15.0), np.deg2rad(-15.0)

angles_list = [
    (k_mag * np.sin(theta1), 0.0),  # Channel 0: Angle 1
    (k_mag * np.sin(theta2), 0.0)   # Channel 1: Angle 2
]


def tv_loss(inputs):
    n, c, h, w = inputs.shape
    grad_x = inputs[:, :, 1:, :] - inputs[:, :, :-1, :]
    grad_y = inputs[:, :, :, 1:] - inputs[:, :, :, :-1]
    tv = (grad_x.abs().sum() + grad_y.abs().sum()) / (n * c * h * w)
    return tv


def main():
    print(f"Initializing GedankenNet Training on Real Experimental Data on {device}...")

    # Check for real BMP hologram pairs in data_raw
    bmp_files = glob.glob(os.path.join(RAW_DIR, '*.bmp')) + glob.glob(os.path.join(RAW_DIR, '*.png'))

    if len(bmp_files) >= 2:
        print(f"Found {len(bmp_files)} real image files in '{RAW_DIR}'. Training directly on REAL experimental holograms!")
        train_dataset = RealHoloTrainDataset(
            RAW_DIR,
            trans=np_transforms.Compose([
                np_transforms.RandomCrop(S),
                np_transforms.RandomHorizontalFlip(),
                np_transforms.ToTensor()
            ])
        )
    else:
        print(f"[Notice] Less than 2 real images found in '{RAW_DIR}'. Falling back to synthetic training mode.")
        train_file_paths = glob.glob(os.path.join(TRAIN_PATH, '*.png')) + glob.glob(os.path.join(TRAIN_PATH, '*.bmp'))
        train_dataset = GedankenOffAxisDataset(
            train_file_paths, angles_list,
            trans=np_transforms.Compose([
                np_transforms.RandomCrop(S),
                np_transforms.RandomHorizontalFlip(),
                np_transforms.ToTensor()
            ]),
            params=params,
            num_samples=500
        )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    path_tag = f"Gedanken_RealOffAxis_ep={epochs}_m={modes}_w={width}"
    path_model = os.path.join('Models', path_tag)
    os.makedirs(path_model, exist_ok=True)
    writer = SummaryWriter(os.path.join("runs", path_tag))

    model = FNO2d(modes, width, in_channel=2, out_channel=1).to(device)
    print(f"Model parameters: {count_params(model)}")

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)

    maeloss = nn.L1Loss(reduction='mean')

    hann_window = torch.outer(torch.hann_window(S), torch.hann_window(S))
    hann_window = torch.fft.ifftshift(hann_window).unsqueeze(0).unsqueeze(0).to(device)

    start_ep = -1
    checkpoint_path = os.path.join(path_model, "checkpoint.pth")
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        start_ep = checkpoint['epoch']
        print(f"Resuming checkpoint from epoch {start_ep + 1}")
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])

    # Batch angle tensor
    angles_batch_tensor = [angles_list for _ in range(batch_size)]

    # Training Loop directly on Real Experimental Holograms
    for ep in range(start_ep + 1, epochs):
        model.train()
        t1 = default_timer()
        train_loss_epoch = 0.0

        for i, batch in enumerate(train_loader):
            if i >= batch_per_ep:
                break

            if isinstance(batch, (list, tuple)):
                xx = batch[0]  # Real holograms [N, 2, H, W]
            else:
                xx = batch

            xx = xx.to(device)
            # Normalize real input holograms by mean intensity
            xx_norm = xx / torch.mean(xx, dim=(2, 3), keepdim=True)

            # Predict phase map from real holograms
            pred_ph, _ = model(xx_norm)  # [N, 1, H, W]

            # Re-simulate holograms via Direct Physics Forward Model
            im_x = batch_forward_direct(pred_ph, angles_batch_tensor, params)  # [N, 2, H, W]

            # Self-Supervised Physics Consistency Loss:
            # Match simulated holograms im_x with real input holograms xx_norm
            loss = 0.0
            loss += maeloss(torch.fft.fft2(im_x) * hann_window, torch.fft.fft2(xx_norm) * hann_window) * 0.1
            loss += maeloss(im_x, xx_norm) * 10.0 + tv_loss(pred_ph) * 5.0

            train_loss_epoch += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        writer.add_scalar('Loss/Train_Real', train_loss_epoch / (i + 1), ep)

        if (ep + 1) % 50 == 0 or ep == epochs - 1:
            torch.save(model, os.path.join(path_model, "final_model.pth"))
            torch.save({
                'epoch': ep,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, checkpoint_path)

        scheduler.step()
        t2 = default_timer()
        print(f"Epoch [{ep+1}/{epochs}] Time: {t2 - t1:.2f}s | Real Physics Consistency Loss: {train_loss_epoch / (i + 1):.4f}")

    torch.save(model, os.path.join(path_model, "final_model.pth"))
    print("Training on Real Data Completed!")


if __name__ == '__main__':
    main()
