"""Optional PyTorch tiled U-Net inference. Import is delayed so torch is optional."""
from pathlib import Path

import numpy as np


def sample_patches(images, masks, size=512, per_image=16, seed=0):
    """Random paired patches for U-Net training; masks may contain optional speck labels."""
    rng = np.random.default_rng(seed); xs = []; ys = []
    for image, mask in zip(images, masks):
        pad_y, pad_x = max(0, size-image.shape[0]), max(0, size-image.shape[1])
        image = np.pad(image, ((0, pad_y), (0, pad_x)), mode="reflect"); mask = np.pad(mask, ((0, pad_y), (0, pad_x)))
        for _ in range(per_image):
            y = rng.integers(0, image.shape[0]-size+1); x = rng.integers(0, image.shape[1]-size+1)
            xs.append(image[y:y+size, x:x+size]); ys.append(mask[y:y+size, x:x+size])
    return np.asarray(xs), np.asarray(ys)


def train_unet(patches, masks, epochs=20, batch_size=8, learning_rate=1e-3):
    """Optional binary U-Net-style trainer: Dice + focal BCE, strong image augmentation."""
    import torch
    import torch.nn as nn
    model = _model(torch, nn); optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    x = torch.as_tensor(patches, dtype=torch.float32)[:, None]; y = torch.as_tensor(masks > 0, dtype=torch.float32)[:, None]
    for _ in range(epochs):
        for indices in torch.randperm(len(x)).split(batch_size):
            image, target = x[indices].clone(), y[indices]
            if torch.rand(()) < .5: image, target = image.flip(-1), target.flip(-1)
            if torch.rand(()) < .5: image, target = image.flip(-2), target.flip(-2)
            if torch.rand(()) < .5: image = 1 - image  # polarity
            image = (image * (0.6 + .8*torch.rand(())) + .15*torch.randn_like(image)).clamp(0, 1)  # brightness/noise
            logits = model(image); prob = logits.sigmoid(); dice = 1-(2*(prob*target).sum()+1)/((prob+target).sum()+1)
            bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none"); focal = ((1-prob).abs()**2*bce).mean()
            optimizer.zero_grad(); (dice+focal).backward(); optimizer.step()
    return model


def _model(torch, nn):
    # Compact comparable segmentation network; replace only when accuracy needs larger U-Net.
    class TinyUNet(nn.Module):
        def __init__(self):
            super().__init__(); self.net = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 1))
        def forward(self, x): return self.net(x)
    return TinyUNet()


def unet_detect(gray: np.ndarray, cfg: dict):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None, "PyTorch unavailable"
    weights = cfg["unet"].get("weights")
    if not weights or not Path(weights).is_file():
        return None, "trained U-Net weights unavailable"

    model = _model(torch, nn).eval(); model.load_state_dict(torch.load(weights, map_location="cpu"))
    tile, overlap = cfg["unet"]["tile_size"], cfg["unet"]["overlap"]; step = tile - overlap
    padded = np.pad(gray, ((0, max(0, tile-gray.shape[0])), (0, max(0, tile-gray.shape[1]))), mode="reflect")
    score = np.zeros_like(padded, float); weight = np.zeros_like(padded, float); window = np.outer(np.hanning(tile), np.hanning(tile)) + 1e-3
    ys = list(range(0, padded.shape[0]-tile+1, step)); xs = list(range(0, padded.shape[1]-tile+1, step))
    if ys[-1] != padded.shape[0]-tile: ys.append(padded.shape[0]-tile)
    if xs[-1] != padded.shape[1]-tile: xs.append(padded.shape[1]-tile)
    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = torch.from_numpy(padded[y:y+tile, x:x+tile]).float()[None, None]
                prediction = torch.sigmoid(model(patch))[0, 0].numpy(); score[y:y+tile, x:x+tile] += prediction*window; weight[y:y+tile, x:x+tile] += window
    score = (score/weight)[:gray.shape[0], :gray.shape[1]]
    from .detectors import Detection
    return Detection(score >= cfg["unet"]["threshold"], score, {"unet_probability": score}), None
