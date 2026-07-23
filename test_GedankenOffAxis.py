###############################################################
#  Test / Inference Script for Direct Off-Axis Holograms (.bmp)
#  - Continuous [sin(phi), cos(phi)] Phase Reconstruction via UNetPhase
#  - Automatic unwrap via atan2(sin, cos)
###############################################################

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import glob

from utilities import device
from my_tools_offaxis import HoloBmpDataset, min_max_norm
import np_transforms

RAW_DIR = 'data_raw'
path_output = os.path.join('outputs', 'Gedanken_DirectOffAxis_BMP')

os.makedirs(path_output, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

# Find trained model dynamically in Models directory
model_candidates = glob.glob('Models/**/final_model.pth', recursive=True) + \
                   glob.glob('Models/**/*.pth', recursive=True)

if len(model_candidates) > 0:
    model_filepath = model_candidates[0]
    print(f"Found trained model: {model_filepath}")
    try:
        model = torch.load(model_filepath, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(model_filepath, map_location=device)
    model.eval()
else:
    print(f"[Warning] No trained model found in 'Models/'. Initializing UNetPhase model structure.")
    from networks.unet_model import UNetPhase
    model = UNetPhase(in_channels=2, out_channels=2).to(device)
    model.eval()

test_dataset = HoloBmpDataset(
    RAW_DIR,
    trans=np_transforms.Compose([
        np_transforms.CenterCrop(512),
        np_transforms.ToTensor()
    ])
)

print(f"Loaded {len(test_dataset)} real BMP samples from '{RAW_DIR}'.")

with torch.no_grad():
    for i in range(len(test_dataset)):
        xx, sample_name = test_dataset[i]
        xx = xx.unsqueeze(0).to(device)  # [1, 2, H, W]

        # Mean normalization across spatial dimensions
        xx_norm = xx / torch.mean(xx, dim=(2, 3), keepdim=True)

        # Predict continuous [sin(phi), cos(phi)] representation
        pred_sc, _ = model(xx_norm)  # [1, 2, H, W]

        # Recover continuous unwrapped phase map: phi = atan2(sin, cos)
        sin_p = pred_sc[:, 0:1, :, :]
        cos_p = pred_sc[:, 1:2, :, :]
        pred_ph = torch.atan2(sin_p, cos_p)  # [1, 1, H, W]

        xx_np = xx.cpu().numpy().squeeze()
        pred_np = pred_ph.cpu().numpy().squeeze()

        # Min-max normalization for output phase visualization
        pred_visual = min_max_norm(pred_np - pred_np.mean())

        # Save output reconstructed phase maps
        plt.imsave(os.path.join(path_output, f"{sample_name}_reconstructed_phase.bmp"), pred_visual, cmap='gray')
        plt.imsave(os.path.join(path_output, f"{sample_name}_reconstructed_phase_color.jpg"), pred_visual, cmap='viridis')

        print(f"[{i+1}/{len(test_dataset)}] Processed: {sample_name} -> Saved to {path_output}")

print(f"Inference finished! Unwrapped phase results saved in '{path_output}'")
