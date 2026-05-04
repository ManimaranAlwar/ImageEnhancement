
import torch
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import numpy as np

# Try modern quantization API, fall back if unavailable
try:
    import torch.ao.quantization as aoq
except Exception:  # pragma: no cover
    aoq = None

# Apply pruning to model to reduce size and improve latency
def apply_pruning(model, pruning_rate=0.4):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=pruning_rate)
    return model

# Remove pruning reparametrization to finalize weights
def remove_pruning(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, 'weight')
            except Exception:
                pass
    return model

# Fuse common Conv+ReLU patterns for better quantization
def fuse_model(model):
    if aoq is None:
        return model
    modules_to_fuse = []
    # Known conv-relu pairs in our model
    for pair in [
        ('conv1', 'relu1'),
        ('conv2', 'relu2'),
        ('conv3', 'relu3'),
        ('conv4', 'relu4'),
        ('conv5', 'relu5'),
        ('conv6', 'relu6'),
    ]:
        if all(hasattr(model, name) for name in pair):
            modules_to_fuse.append(list(pair))
    if modules_to_fuse:
        try:
            aoq.fuse_modules(model, modules_to_fuse, inplace=True)
        except Exception:
            pass
    return model

# Prepare model for Quantization-Aware Training (QAT)
def prepare_qat_model(model):
    if aoq is None:
        return model
    fuse_model(model)
    try:
        qconfig = aoq.get_default_qat_qconfig('fbgemm')
    except Exception:
        qconfig = aoq.get_default_qat_qconfig('qnnpack')
    model.qconfig = qconfig
    # Exclude transformer and attention-related submodules from QAT (not supported by default)
    for name, mod in model.named_modules():
        if 'transformer' in name or isinstance(mod, torch.nn.MultiheadAttention) or isinstance(mod, torch.nn.LayerNorm) or isinstance(mod, torch.nn.Linear):
            # Be conservative: disable QAT for transformer path to avoid conversion assertion
            mod.qconfig = None
    model = aoq.prepare_qat(model, inplace=True)
    return model

# Convert a QAT-prepared model to quantized

def convert_qat_model(model):
    if aoq is None:
        return model
    model.eval()
    try:
        model = aoq.convert(model, inplace=True)
    except Exception:
        pass
    return model

# Back-compat shim used by older code paths
def mixed_precision_quantization(model):
    return convert_qat_model(model)

# Synthetic corruption operators used for corruption-aware losses
def apply_corruption(image_tensor, corruption_types=('noise', 'blur', 'low-light')):
    """
    image_tensor: (B, C, H, W) in [0,1]
    Returns a corrupted clone with specified types applied sequentially.
    """
    x = image_tensor.clone()
    B, C, H, W = x.shape
    device = x.device
    for c in corruption_types:
        if c == 'noise':
            noise = torch.randn_like(x) * 0.05
            x = (x + noise).clamp(0.0, 1.0)
        elif c == 'blur':
            # Simple 3x3 box blur via depthwise conv
            kernel = torch.ones((C, 1, 3, 3), device=device) / 9.0
            x = torch.nn.functional.conv2d(x, kernel, padding=1, groups=C)
        elif c == 'low-light':
            x = (x * 0.5).clamp(0.0, 1.0)
    return x

# Blind-spot mask generation

def make_blindspot_mask(batch_size, height, width, ratio=0.1, device='cpu'):
    """Generate a binary mask (B,1,H,W) with given masked pixel ratio."""
    total = height * width
    k = max(1, int(total * ratio))
    mask = torch.zeros((batch_size, 1, height, width), device=device)
    # For each sample, randomly select k pixels to mask
    for b in range(batch_size):
        idx = torch.randperm(total, device=device)[:k]
        mask.view(batch_size, -1)[b, idx] = 1.0
    return mask

# Self-supervised loss with blind-spot reconstruction and corruption consistency

def self_supervised_loss(output, target, mask, input_image=None, perceptual=None):
    """
    output, target, input_image: (B,C,H,W) in [0,1]
    mask: (B,1,H,W) with 1 on masked pixels where reconstruction is enforced.
    Loss terms:
      - Blind-spot reconstruction on masked pixels (MSE)
      - Identity on unmasked region to preserve content
      - Corruption-aware consistency between output and target under same corruption
      - Optional perceptual loss (on masked region)
    """
    eps = 1e-8
    B, C, H, W = output.shape
    masked = mask
    unmasked = 1.0 - masked

    # Reconstruction loss on masked pixels
    diff_masked = (output - target) * masked
    mse_masked = (diff_masked.pow(2).sum() / (masked.sum() * C + eps))

    # Identity loss on unmasked pixels to avoid unnecessary changes
    if input_image is None:
        input_image = target
    diff_unmasked = (output - input_image) * unmasked
    mse_unmasked = (diff_unmasked.pow(2).sum() / (unmasked.sum() * C + eps))

    # Corruption-aware consistency
    out_c = apply_corruption(output)
    tgt_c = apply_corruption(target)
    mse_corr = F.mse_loss(out_c, tgt_c)

    # Optional perceptual on masked region
    perc = 0.0
    if perceptual is not None:
        # Masked region via weighting
        perc = perceptual(output * masked, target * masked)

    # Weights tuned for stability on CPU-only edge targets
    loss = 0.6 * mse_masked + 0.2 * mse_unmasked + 0.15 * mse_corr + 0.05 * perc
    return loss

# Efficient SSIM loss (PyTorch implementation)
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        from torchvision import models
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.device = device
    def forward(self, output, target):
        # output, target: (B, C, H, W), range [0, 1]
        # VGG expects 3-channel, 224x224, normalized
        def preprocess(img):
            img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
            mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1,3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1,3,1,1)
            return (img - mean) / std
        output_vgg = self.vgg(preprocess(output))
        target_vgg = self.vgg(preprocess(target))
        return F.l1_loss(output_vgg, target_vgg)

# Combined loss function (supervised) retained for backward compatibility

def ssim_loss(output, target):
    # output, target: (B, C, H, W), range [0, 1]
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = F.avg_pool2d(output, 3, 1)
    mu_y = F.avg_pool2d(target, 3, 1)
    sigma_x = F.avg_pool2d(output ** 2, 3, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(target ** 2, 3, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(output * target, 3, 1) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return 1 - ssim_map.mean()


def final_loss(output, target, input, perceptual=None):
    mse = F.mse_loss(output, target)
    ssim = ssim_loss(output, target)
    perceptual_loss = perceptual(output, target) if perceptual is not None else 0.0
    return 0.6 * mse + 0.3 * ssim + 0.1 * perceptual_loss

# Efficient SSIM loss (PyTorch implementation)
def ssim_loss(output, target):
    # output, target: (B, C, H, W), range [0, 1]
    from torch import nn
    from torch import mean, var
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = F.avg_pool2d(output, 3, 1)
    mu_y = F.avg_pool2d(target, 3, 1)
    sigma_x = F.avg_pool2d(output ** 2, 3, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(target ** 2, 3, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(output * target, 3, 1) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return 1 - ssim_map.mean()

# Efficient VGG-based perceptual loss
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        from torchvision import models
        try:
            # Load VGG16 weights from local file
            base = models.vgg16(weights=None)
            base.load_state_dict(torch.load('vgg16-397923af.pth'))
        except Exception:
            base = models.vgg16(weights=None)
        vgg = base.features[:16].eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.device = device
    def forward(self, output, target):
        # output, target: (B, C, H, W), range [0, 1]
        # VGG expects 3-channel, 224x224, normalized
        def preprocess(img):
            img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
            mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1,3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1,3,1,1)
            return (img - mean) / std
        output_vgg = self.vgg(preprocess(output))
        target_vgg = self.vgg(preprocess(target))
        return F.l1_loss(output_vgg, target_vgg)

# Combined loss function
def final_loss(output, target, input, perceptual=None):
    mse = F.mse_loss(output, target)
    ssim = ssim_loss(output, target)
    perceptual_loss = perceptual(output, target) if perceptual is not None else 0.0
    return 0.6 * mse + 0.3 * ssim + 0.1 * perceptual_loss

# Post-processing for output image (e.g., denormalizing, clipping)
def post_process_output(output):
    output = output.squeeze(0).detach().cpu().numpy()
    output = np.transpose(output, (1, 2, 0)) * 255.0  # Denormalize
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output
