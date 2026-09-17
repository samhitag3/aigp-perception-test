"""Load-time data augmentations for GateNet (MonoRace recipe).

Applied stochastically per sample during training:
  * randomized affine: rotation, translation, scaling (image + mask),
  * HSV photometric jitter (hue/sat/value) — image only,
  * artificial motion blur: square averaging kernel, random size 5-15 px and
    random orientation — image only,
  * additive Gaussian ("thermal") noise — image only.

Images are HxWx3 uint8 RGB; masks are HxW float32 in {0,1}.
Geometric transforms use the SAME warp matrix for image and mask (nearest for
the mask so it stays binary).
"""

from __future__ import annotations

import cv2
import numpy as np


def _rand(lo, hi):
    return float(np.random.uniform(lo, hi))


class Augment:
    def __init__(
        self,
        train: bool = True,
        rotate_deg: float = 15.0,
        translate: float = 0.1,        # fraction of image size
        scale=(0.8, 1.2),
        hsv=(10.0, 0.3, 0.3),          # (hue deg, sat frac, val frac)
        motion_blur=(5, 15),           # kernel size range (px); 0 disables
        motion_blur_p: float = 0.5,
        gauss_noise_std: float = 12.0, # on 0-255 scale
        gauss_noise_p: float = 0.5,
        hflip_p: float = 0.0,          # off by default (gate L/R asymmetry)
    ):
        self.train = train
        self.rotate_deg = rotate_deg
        self.translate = translate
        self.scale = scale
        self.hsv = hsv
        self.motion_blur = motion_blur
        self.motion_blur_p = motion_blur_p
        self.gauss_noise_std = gauss_noise_std
        self.gauss_noise_p = gauss_noise_p
        self.hflip_p = hflip_p

    # ---- geometric -------------------------------------------------------
    def _affine(self, img, mask):
        h, w = img.shape[:2]
        angle = _rand(-self.rotate_deg, self.rotate_deg)
        sc = _rand(*self.scale)
        tx = _rand(-self.translate, self.translate) * w
        ty = _rand(-self.translate, self.translate) * h
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, sc)
        M[0, 2] += tx
        M[1, 2] += ty
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img, mask

    # ---- photometric (image only) ---------------------------------------
    def _hsv(self, img):
        dh, ds, dv = self.hsv
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + _rand(-dh, dh)) % 180.0
        hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + _rand(-ds, ds)), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 + _rand(-dv, dv)), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def _motion_blur(self, img):
        lo, hi = self.motion_blur
        k = int(np.random.randint(lo, hi + 1))
        if k < 3:
            return img
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k             # horizontal line kernel
        angle = _rand(0, 180)
        M = cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (k, k))
        s = kernel.sum()
        if s > 1e-6:
            kernel /= s
        return cv2.filter2D(img, -1, kernel)

    def _gauss_noise(self, img):
        noise = np.random.normal(0.0, self.gauss_noise_std, img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # ---- entry point -----------------------------------------------------
    def __call__(self, img: np.ndarray, mask: np.ndarray):
        if not self.train:
            return img, mask
        if self.hflip_p and np.random.rand() < self.hflip_p:
            img = img[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
        img, mask = self._affine(img, mask)
        img = self._hsv(img)
        if np.random.rand() < self.motion_blur_p:
            img = self._motion_blur(img)
        if np.random.rand() < self.gauss_noise_p:
            img = self._gauss_noise(img)
        return img, mask


def build_augment(cfg: dict | None, train: bool = True) -> Augment:
    cfg = dict(cfg or {})
    cfg["train"] = train
    return Augment(**cfg)
