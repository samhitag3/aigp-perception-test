from __future__ import annotations
import math
import cv2
import numpy as np


def _sample_range(rng, x):
    return float(rng.uniform(float(x[0]), float(x[1])))


def sample_sequence_noise(rng: np.random.Generator, noise_cfg: dict) -> dict:
    names = list(noise_cfg["tier_probabilities"].keys())
    p = np.array([noise_cfg["tier_probabilities"][n] for n in names], dtype=float)
    p /= p.sum()
    tier = str(rng.choice(names, p=p))
    ranges = noise_cfg["tiers"][tier]
    params = {k: _sample_range(rng, v) for k, v in ranges.items()}
    params["tier"] = tier
    params["flare_x"] = float(rng.uniform(0.05, 0.95))
    params["flare_y"] = float(rng.uniform(0.05, 0.55))
    params["shadow_angle"] = float(rng.uniform(0, 2 * np.pi))
    params["phase"] = float(rng.uniform(0, 2 * np.pi))
    return params


def _motion_blur(img, k: int, angle_deg: float):
    k = int(max(0, round(k)))
    if k < 3:
        return img
    if k % 2 == 0:
        k += 1
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle_deg, 1.0)
    ker = cv2.warpAffine(ker, M, (k, k))
    ker /= max(ker.sum(), 1e-9)
    return cv2.filter2D(img, -1, ker)


def apply_photometric(img_rgb: np.ndarray, params: dict, frame_idx: int, nframes: int,
                      rng: np.random.Generator) -> np.ndarray:
    x = img_rgb.astype(np.float32) / 255.0
    t = frame_idx / max(nframes - 1, 1)
    slow = 1.0 + 0.05 * np.sin(2 * np.pi * t + params["phase"])

    x *= params["brightness"] * slow
    mean = x.mean(axis=(0, 1), keepdims=True)
    x = (x - mean) * params["contrast"] + mean
    x = np.clip(x, 0, 1)
    gamma = max(params["gamma"], 1e-3)
    x = np.power(x, 1.0 / gamma)

    hsv = cv2.cvtColor(np.clip(x * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= params["saturation"]
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    temp = params["color_temp_shift"]
    x[..., 0] *= (1.0 + temp)
    x[..., 2] *= (1.0 - temp)

    h, w = x.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = (xx - w / 2) / (w / 2)
    yn = (yy - h / 2) / (h / 2)
    r2 = xn * xn + yn * yn
    x *= np.clip(1.0 - params["vignette_strength"] * r2, 0.35, 1.0)[..., None]

    ang = params["shadow_angle"]
    proj = xn * np.cos(ang) + yn * np.sin(ang) + 0.2 * np.sin(2 * np.pi * t + params["phase"])
    shadow = 1.0 - params["shadow_strength"] * np.clip((proj + 0.4) / 0.8, 0, 1)
    x *= shadow[..., None]

    if params["flare_strength"] > 1e-4:
        cx = (params["flare_x"] + 0.08 * np.sin(2 * np.pi * t + params["phase"])) * w
        cy = (params["flare_y"] + 0.05 * np.cos(2 * np.pi * t + params["phase"])) * h
        rr = ((xx - cx) ** 2 + (yy - cy) ** 2) / max((0.22 * w) ** 2, 1.0)
        flare = np.exp(-rr) * params["flare_strength"]
        x = x + flare[..., None]

    x = np.clip(x, 0, 1)
    out = (x * 255).astype(np.uint8)

    sigma = params["gaussian_blur_sigma"]
    if sigma > 0.05:
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma, sigmaY=sigma)
    out = _motion_blur(out, int(params["motion_blur_px"]), angle_deg=7.0 * np.sin(2 * np.pi * t + params["phase"]))

    y = out.astype(np.float32)
    if params["gaussian_noise_std"] > 0:
        y += rng.normal(0, params["gaussian_noise_std"], y.shape).astype(np.float32)
    if params["speckle_std"] > 0:
        y *= (1.0 + rng.normal(0, params["speckle_std"], y.shape).astype(np.float32))
    y = np.clip(y, 0, 255).astype(np.uint8)

    q = int(np.clip(round(params["jpeg_quality"]), 20, 100))
    bgr = cv2.cvtColor(y, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        y = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return y


def _gaussian_kernel1d_torch(sigma: float, device):
    import torch
    sigma = max(float(sigma), 1e-4)
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(x * x) / (2 * sigma * sigma))
    k = k / k.sum()
    return k


def _apply_separable_gaussian_torch(x, sigma: float):
    import torch
    import torch.nn.functional as F
    if sigma <= 0.05:
        return x
    B, C, H, W = x.shape
    k1 = _gaussian_kernel1d_torch(sigma, x.device)
    kx = k1.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    ky = k1.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    pad = (k1.numel() - 1) // 2
    x = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, kx, groups=C)
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    x = F.conv2d(x, ky, groups=C)
    return x


def _make_motion_kernel_torch(k: int, angle_deg: float, device):
    import torch
    k = int(max(0, round(k)))
    if k < 3:
        return None
    if k % 2 == 0:
        k += 1
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(angle_deg), 1.0)
    ker = cv2.warpAffine(ker, M, (k, k))
    ker /= max(float(ker.sum()), 1e-9)
    return torch.from_numpy(ker).to(device=device, dtype=torch.float32)


def _apply_motion_blur_torch(x, k: int, angle_deg: float):
    import torch.nn.functional as F
    ker = _make_motion_kernel_torch(k, angle_deg, x.device)
    if ker is None:
        return x
    B, C, H, W = x.shape
    w = ker.view(1, 1, ker.shape[0], ker.shape[1]).repeat(C, 1, 1, 1)
    pad = (ker.shape[0] - 1) // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    x = F.conv2d(x, w, groups=C)
    return x


def apply_photometric_batch(imgs_rgb: list[np.ndarray], params: dict, frame_indices: list[int],
                            nframes: int, rng_seed: int = 0, device: str = "cuda") -> list[np.ndarray]:
    """GPU-accelerated batched photometric augmentation.

    Falls back to per-frame CPU augmentation if torch/CUDA is unavailable.
    """
    try:
        import torch
    except Exception:
        outs = []
        for i, img in zip(frame_indices, imgs_rgb):
            rng = np.random.default_rng(rng_seed + 7919 * int(i) + 17)
            outs.append(apply_photometric(img, params, int(i), int(nframes), rng))
        return outs

    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        outs = []
        for i, img in zip(frame_indices, imgs_rgb):
            rng = np.random.default_rng(rng_seed + 7919 * int(i) + 17)
            outs.append(apply_photometric(img, params, int(i), int(nframes), rng))
        return outs

    x_np = np.stack(imgs_rgb, axis=0)
    B, H, W, C = x_np.shape
    x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32) / 255.0
    x = x.permute(0, 3, 1, 2).contiguous()

    t = torch.tensor(frame_indices, device=device, dtype=torch.float32) / max(float(nframes - 1), 1.0)
    phase = float(params["phase"])
    slow = 1.0 + 0.05 * torch.sin(2 * math.pi * t + phase)

    x = x * (float(params["brightness"]) * slow.view(B, 1, 1, 1))
    mean = x.mean(dim=(2, 3), keepdim=True)
    x = (x - mean) * float(params["contrast"]) + mean
    x = torch.clamp(x, 0.0, 1.0)
    gamma = max(float(params["gamma"]), 1e-3)
    x = torch.pow(x, 1.0 / gamma)

    # saturation approximation in RGB using per-pixel grayscale interpolation
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    sat = float(params["saturation"])
    x = gray + sat * (x - gray)

    temp = float(params["color_temp_shift"])
    x[:, 0] *= (1.0 + temp)
    x[:, 2] *= (1.0 - temp)

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    xn = (xx - W / 2) / max(W / 2, 1.0)
    yn = (yy - H / 2) / max(H / 2, 1.0)
    r2 = xn * xn + yn * yn
    vignette = torch.clamp(1.0 - float(params["vignette_strength"]) * r2, 0.35, 1.0)
    x = x * vignette.view(1, 1, H, W)

    ang = float(params["shadow_angle"])
    proj = xn * math.cos(ang) + yn * math.sin(ang) + 0.2 * torch.sin(2 * math.pi * t + phase).view(B, 1, 1)
    shadow = 1.0 - float(params["shadow_strength"]) * torch.clamp((proj + 0.4) / 0.8, 0.0, 1.0)
    x = x * shadow.view(B, 1, H, W)

    flare_strength = float(params["flare_strength"])
    if flare_strength > 1e-4:
        cx = (float(params["flare_x"]) + 0.08 * torch.sin(2 * math.pi * t + phase)) * W
        cy = (float(params["flare_y"]) + 0.05 * torch.cos(2 * math.pi * t + phase)) * H
        flare_maps = []
        denom = max((0.22 * W) ** 2, 1.0)
        for bi in range(B):
            rr = ((xx - cx[bi]) ** 2 + (yy - cy[bi]) ** 2) / denom
            flare_maps.append(torch.exp(-rr) * flare_strength)
        flare = torch.stack(flare_maps, dim=0).unsqueeze(1)
        x = x + flare

    x = torch.clamp(x, 0.0, 1.0)
    x = _apply_separable_gaussian_torch(x, float(params["gaussian_blur_sigma"]))

    k = int(round(float(params["motion_blur_px"])))
    if k >= 3:
        angles = 7.0 * torch.sin(2 * math.pi * t + phase)
        # Group frames by rounded angle to avoid per-frame convolutions when possible.
        out_chunks = []
        used = torch.zeros(B, device=device, dtype=torch.bool)
        for ang_rounded in torch.unique(torch.round(angles)).tolist():
            idx = torch.where(torch.round(angles) == ang_rounded)[0]
            if idx.numel() == 0:
                continue
            blurred = _apply_motion_blur_torch(x[idx], k, float(ang_rounded))
            out_chunks.append((idx, blurred))
            used[idx] = True
        x2 = torch.empty_like(x)
        if out_chunks:
            for idx, chunk in out_chunks:
                x2[idx] = chunk
            if not bool(torch.all(used)):
                x2[~used] = x[~used]
            x = x2

    g = torch.Generator(device=device)
    g.manual_seed(int(rng_seed))
    noise_std = float(params["gaussian_noise_std"]) / 255.0
    if noise_std > 0:
        x = x + torch.randn(x.shape, device=device, generator=g) * noise_std
    speckle_std = float(params["speckle_std"])
    if speckle_std > 0:
        x = x * (1.0 + torch.randn(x.shape, device=device, generator=g) * speckle_std)
    x = torch.clamp(x, 0.0, 1.0)

    # JPEG artifact simulation still uses CPU encode/decode.
    y = (x.permute(0, 2, 3, 1).contiguous().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    q = int(np.clip(round(float(params["jpeg_quality"])), 20, 100))
    outs = []
    for frame in y:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        outs.append(frame)
    return outs
