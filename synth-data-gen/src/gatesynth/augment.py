from __future__ import annotations
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
    params["shadow_angle"] = float(rng.uniform(0, 2*np.pi))
    params["phase"] = float(rng.uniform(0, 2*np.pi))
    return params


def _motion_blur(img, k: int, angle_deg: float):
    k = int(max(0, round(k)))
    if k < 3:
        return img
    if k % 2 == 0:
        k += 1
    ker = np.zeros((k, k), np.float32)
    ker[k//2, :] = 1.0
    M = cv2.getRotationMatrix2D((k/2 - 0.5, k/2 - 0.5), angle_deg, 1.0)
    ker = cv2.warpAffine(ker, M, (k, k))
    ker /= max(ker.sum(), 1e-9)
    return cv2.filter2D(img, -1, ker)


def apply_photometric(img_rgb: np.ndarray, params: dict, frame_idx: int, nframes: int,
                      rng: np.random.Generator) -> np.ndarray:
    x = img_rgb.astype(np.float32) / 255.0
    t = frame_idx / max(nframes - 1, 1)
    slow = 1.0 + 0.05 * np.sin(2*np.pi*t + params["phase"])

    # brightness / contrast / gamma are sequence-fixed strengths with small smooth temporal drift
    x *= params["brightness"] * slow
    mean = x.mean(axis=(0,1), keepdims=True)
    x = (x - mean) * params["contrast"] + mean
    x = np.clip(x, 0, 1)
    gamma = max(params["gamma"], 1e-3)
    x = np.power(x, 1.0 / gamma)

    hsv = cv2.cvtColor(np.clip(x*255,0,255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[...,1] *= params["saturation"]
    hsv[...,1] = np.clip(hsv[...,1],0,255)
    x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    # color temperature proxy
    temp = params["color_temp_shift"]
    x[...,0] *= (1.0 + temp)
    x[...,2] *= (1.0 - temp)

    h, w = x.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = (xx - w/2)/(w/2)
    yn = (yy - h/2)/(h/2)
    r2 = xn*xn + yn*yn
    x *= np.clip(1.0 - params["vignette_strength"]*r2, 0.35, 1.0)[...,None]

    # broad moving shadow
    ang = params["shadow_angle"]
    proj = xn*np.cos(ang) + yn*np.sin(ang) + 0.2*np.sin(2*np.pi*t + params["phase"])
    shadow = 1.0 - params["shadow_strength"] * np.clip((proj + 0.4)/0.8, 0, 1)
    x *= shadow[...,None]

    # flare/glare blob
    if params["flare_strength"] > 1e-4:
        cx = (params["flare_x"] + 0.08*np.sin(2*np.pi*t + params["phase"])) * w
        cy = (params["flare_y"] + 0.05*np.cos(2*np.pi*t + params["phase"])) * h
        rr = ((xx-cx)**2 + (yy-cy)**2) / max((0.22*w)**2, 1.0)
        flare = np.exp(-rr) * params["flare_strength"]
        x = x + flare[...,None]

    x = np.clip(x,0,1)
    out = (x*255).astype(np.uint8)

    sigma = params["gaussian_blur_sigma"]
    if sigma > 0.05:
        out = cv2.GaussianBlur(out, (0,0), sigmaX=sigma, sigmaY=sigma)
    out = _motion_blur(out, int(params["motion_blur_px"]), angle_deg=7.0*np.sin(2*np.pi*t + params["phase"]))

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
