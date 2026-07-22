###############################################################
#  Test / Inference Script for Direct Off-Axis Holograms (.bmp)
#  Reads real BMP images from folder /data_raw
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
    model = torch.load(model_filepath, map_location=device)
    model.eval()
else:
    print(f"[Warning] No trained model found in 'Models/'. Initializing untrained model structure.")
    from networks.fno import FNO2d
    model = FNO2d(modes=256, width=4, in_channel=2, out_channel=1).to(device)
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

        # Predict phase map
        pred_ph, _ = model(xx_norm)  # [1, 1, H, W]

        xx_np = xx.cpu().numpy().squeeze()
        pred_np = pred_ph.cpu().numpy().squeeze()

        # Min-max normalization for output phase visualization
        pred_visual = min_max_norm(pred_np - pred_np.mean())

        # Save output reconstructed phase and holograms
        plt.imsave(os.path.join(path_output, f"{sample_name}_reconstructed_phase.bmp"), pred_visual, cmap='gray')
        plt.imsave(os.path.join(path_output, f"{sample_name}_reconstructed_phase_color.jpg"), pred_visual, cmap='viridis')

        print(f"[{i+1}/{len(test_dataset)}] Processed: {sample_name} -> Saved to {path_output}")

print(f"Inference finished! Results saved in '{path_output}'")
