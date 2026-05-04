import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from model import CNNTransformerModel
from dataset import CustomImageDataset
from utils import post_process_output
import cv2

# Validation dataset
import random
val_dataset = CustomImageDataset('d:/saveetha/SIDD_Small_sRGB_Only/Data', corruption_types=['noise', 'blur', 'low-light'])
# Select random samples
num_samples = 10  # Number of random samples to test
indices = random.sample(range(len(val_dataset)), min(num_samples, len(val_dataset)))
val_loader = DataLoader(val_dataset, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(indices))

model = CNNTransformerModel()

# Helper: recursively dequantize any quantized tensors in a state_dict-like mapping
def _dequantize_state_dict(sd):
    new_sd = {}
    for k, v in sd.items():
        # If it's a Quantized Tensor (torch._utils.typed_storage or QuantizedTensor), try dequantize
        try:
            if hasattr(v, 'dequantize'):
                new_sd[k] = v.dequantize()
            else:
                new_sd[k] = v
        except Exception:
            new_sd[k] = v
    return new_sd

# Load checkpoint and dequantize parameters if necessary
checkpoint = torch.load('enhancement_model.pth')
if isinstance(checkpoint, dict) and ('state_dict' in checkpoint or 'model_state_dict' in checkpoint):
    # Support common checkpoint wrappers
    key = 'state_dict' if 'state_dict' in checkpoint else 'model_state_dict'
    raw_sd = checkpoint[key]
    raw_sd = _dequantize_state_dict(raw_sd)
    checkpoint[key] = raw_sd
    state_dict = raw_sd
else:
    # Plain state_dict
    state_dict = _dequantize_state_dict(checkpoint) if isinstance(checkpoint, dict) else checkpoint

model.load_state_dict(state_dict, strict=False)  # Ignore missing/unexpected keys
model.eval()

psnr_list = []
ssim_list = []
loss_list = []

# Simple PSNR calculation
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# Simple SSIM calculation (using skimage if available)
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    def ssim(img1, img2):
        return np.nan

import time
proc_t0 = time.process_time()
wall_t0 = time.perf_counter()
latencies_ms = []
for i, batch in enumerate(val_loader):
    corrupted, clean = batch
    with torch.no_grad():
        t0 = time.perf_counter()
        output = model(corrupted)
        t1 = time.perf_counter()
    latencies_ms.append((t1 - t0) * 1000.0)
    output_np = output.squeeze(0).detach().cpu().numpy()
    clean_np = clean.squeeze(0).detach().cpu().numpy()
    # Convert to HWC
    output_np = np.transpose(output_np, (1,2,0))
    clean_np = np.transpose(clean_np, (1,2,0))
    # Calculate metrics
    psnr_val = psnr(output_np, clean_np)
    psnr_list.append(psnr_val)
    ssim_val = ssim(output_np, clean_np, win_size=7, channel_axis=2, data_range=1.0)
    ssim_list.append(ssim_val)
    # Calculate loss (MSE)
    loss = np.mean((output_np - clean_np) ** 2)
    loss_list.append(loss)
    # Optionally save sample results
    cv2.imwrite(f'sample_{i}_corrupted.png', cv2.cvtColor((corrupted.squeeze(0).numpy().transpose(1,2,0)*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    # cv2.imwrite(f'sample_{i}_enhanced.png', cv2.cvtColor((output_np*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'sample_{i}_clean.png', cv2.cvtColor((clean_np*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

# Plot validation loss graph
plt.figure()
plt.plot(loss_list, label='Validation Loss (MSE)')
plt.xlabel('Sample')
plt.ylabel('Loss')
plt.title('Validation Loss per Sample')
plt.legend()
plt.savefig('validation_loss_graph.png')
plt.close()

# Print test results
avg_psnr = float(np.mean(psnr_list))
avg_ssim = float(np.mean(ssim_list))
avg_loss = float(np.mean(loss_list))
avg_latency_ms = float(np.mean(latencies_ms)) if latencies_ms else float('nan')
wall_t1 = time.perf_counter()
proc_t1 = time.process_time()
# Simple power proxy: CPU time per second of wall time (unitless, higher ~ more CPU-bound)
power_proxy = (proc_t1 - proc_t0) / max(1e-9, (wall_t1 - wall_t0))

print(f'Average PSNR: {avg_psnr:.2f}')
print(f'Average SSIM: {avg_ssim:.4f}')
print(f'Average Validation Loss: {avg_loss:.6f}')
print(f'Average Latency: {avg_latency_ms:.2f} ms/frame')
print(f'Power Proxy (CPU_time/Wall_time): {power_proxy:.3f}')

# Save results to file
with open('test_results.txt', 'w') as f:
    f.write(f'Average PSNR: {avg_psnr:.2f}\n')
    f.write(f'Average SSIM: {avg_ssim:.4f}\n')
    f.write(f'Average Validation Loss: {avg_loss:.6f}\n')
    f.write(f'Average Latency (ms/frame): {avg_latency_ms:.2f}\n')
    f.write(f'Power Proxy (CPU_time/Wall_time): {power_proxy:.3f}\n')
