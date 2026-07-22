###############################################################
#  GedankenNet-Phase for Direct Off-Axis Holography
#  Self-Supervised Physics-Consistency Training (NO Wave Propagation)
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
from my_tools_offaxis import batch_forward_direct, GedankenOffAxisDataset
import np_transforms
from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)

# ==============================================================
# CONFIGURATIONS
# ==============================================================
TRAIN_PATH = 'data/train'
VALID_PATH = 'data/valid'

S = 512  # Patch size (512x512)
modes = 256
width = 4

batch_size = 1
batch_per_ep = 250
epochs = 10000
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
    print(f"Initializing Direct Off-Axis Training on {device}...")

    train_file_paths = glob.glob(os.path.join(TRAIN_PATH, '*.png')) + glob.glob(os.path.join(TRAIN_PATH, '*.bmp'))
    if len(train_file_paths) == 0:
        print(f"[Notice] Place background/training images in '{TRAIN_PATH}'.")

    train_dataset = GedankenOffAxisDataset(
        train_file_paths, angles_list,
        np_transforms.Compose([
            np_transforms.RandomCrop(S),
            np_transforms.RandomHorizontalFlip(),
            np_transforms.ToTensor()
        ]),
        params
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    valid_file_paths = glob.glob(os.path.join(VALID_PATH, '*.png')) + glob.glob(os.path.join(VALID_PATH, '*.bmp'))
    valid_dataset = GedankenOffAxisDataset(
        valid_file_paths, angles_list,
        np_transforms.Compose([
            np_transforms.RandomCrop(S),
            np_transforms.RandomHorizontalFlip(),
            np_transforms.ToTensor()
        ]),
        params
    )
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    path_tag = f"Gedanken_DirectOffAxis_ep={epochs}_m={modes}_w={width}"
    path_model = os.path.join('Models', path_tag)
    os.makedirs(path_model, exist_ok=True)
    writer = SummaryWriter(os.path.join("runs", path_tag))

    model = FNO2d(modes, width, in_channel=2, out_channel=1).to(device)
    print(f"Model parameters: {count_params(model)}")

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)

    maeloss = nn.L1Loss(reduction='mean')
    mseloss = nn.MSELoss()

    hann_window = torch.outer(torch.hann_window(S), torch.hann_window(S))
    hann_window = torch.fft.ifftshift(hann_window).unsqueeze(0).unsqueeze(0).to(device)

    start_ep = -1
    min_valid_rmse = 1.0
    checkpoint_path = os.path.join(path_model, "checkpoint.pth")
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        start_ep = checkpoint['epoch']
        print(f"Resuming checkpoint from epoch {start_ep + 1}")
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])

    for ep in range(start_ep + 1, epochs):
        model.train()
        t1 = default_timer()
        train_l2_step = 0.0

        for i, (xx, yy, angles_b) in enumerate(train_loader):
            if i >= batch_per_ep:
                break

            xx = xx.to(device)  # [N, 2, H, W] Input 2-angle holograms
            yy = yy.to(device)  # [N, 1, H, W]

            pred_ph, _ = model(xx)  # [N, 1, H, W]

            # Direct Interference Physics Forward Model (NO wave propagation)
            im_x = batch_forward_direct(pred_ph, angles_b, params)  # [N, 2, H, W]

            loss = 0.0
            loss += maeloss(torch.fft.fft2(im_x) * hann_window, torch.fft.fft2(xx) * hann_window) * 0.1
            loss += maeloss(im_x, xx) * 10.0 + tv_loss(pred_ph) * 5.0

            train_l2_step += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        valid_mse_step = 0.0
        model.eval()
        with torch.no_grad():
            for i, (xx, yy, angles_b) in enumerate(valid_loader):
                xx = xx.to(device)
                yy = yy.to(device)

                pred_ph, _ = model(xx)
                loss = mseloss(pred_ph, yy)
                valid_mse_step += loss.item()

                if i >= 5:
                    break

        writer.add_scalar('Loss/Train', train_l2_step / (i + 1), ep)
        writer.add_scalar('Loss/Valid_MSE', valid_mse_step / (i + 1), ep)

        if valid_mse_step / (i + 1) < min_valid_rmse and ep > 50:
            torch.save(model, os.path.join(path_model, f"ep_{ep}.pth"))
            min_valid_rmse = valid_mse_step / (i + 1)

        if (ep + 1) % 50 == 0:
            torch.save({
                'epoch': ep,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, checkpoint_path)

        scheduler.step()
        t2 = default_timer()
        print(f"Epoch [{ep+1}/{epochs}] Time: {t2 - t1:.2f}s | Train Loss: {train_l2_step:.4f} | Valid MSE: {valid_mse_step:.4f}")

    torch.save(model, os.path.join(path_model, "final_model.pth"))
    print("Training Complete!")


if __name__ == '__main__':
    main()
