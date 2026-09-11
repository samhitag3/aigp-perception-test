from __future__ import annotations
import random
import io
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn.functional as F


class SynchronizedAugment:
    """Pose-safe augmentations: crop/zoom/resize + photometric degradation.

    Crop/resize changes the image sampling model, so K/keypoints/boxes/masks are updated.
    Camera-relative 3D pose is unchanged. Arbitrary 2D rotation is intentionally omitted.
    """
    def __init__(self, cfg: dict, out_hw: tuple[int, int]):
        self.cfg = cfg or {}
        self.out_h, self.out_w = out_hw

    def __call__(self, image: Image.Image, mask: np.ndarray, frame: dict):
        image, mask, frame = self._crop_resize(image, mask, frame)
        image = self._photo(image)
        return image, mask, frame

    def _crop_resize(self, image, mask, frame):
        W, H = image.size
        scale_min = float(self.cfg.get("crop_scale_min", 1.0))
        scale_max = float(self.cfg.get("crop_scale_max", 1.0))
        if scale_min < 1.0 or scale_max < 1.0:
            s = random.uniform(scale_min, scale_max)
            cw, ch = max(2, int(W*s)), max(2, int(H*s))
            left = random.randint(0, max(0, W-cw)); top = random.randint(0, max(0, H-ch))
        else:
            cw, ch, left, top = W, H, 0, 0
        image = image.crop((left, top, left+cw, top+ch)).resize((self.out_w, self.out_h), Image.BILINEAR)
        mask_img = Image.fromarray(mask.astype(np.uint16))
        mask = np.array(mask_img.crop((left, top, left+cw, top+ch)).resize((self.out_w, self.out_h), Image.NEAREST), dtype=np.uint16)
        sx, sy = self.out_w/cw, self.out_h/ch
        K = frame["camera"]["intrinsics"]
        K["fx"] = float(K["fx"]*sx); K["fy"] = float(K["fy"]*sy)
        K["cx"] = float((K["cx"]-left)*sx); K["cy"] = float((K["cy"]-top)*sy)
        K["K"] = [[K["fx"],0.0,K["cx"]],[0.0,K["fy"],K["cy"]],[0.0,0.0,1.0]]
        for g in frame.get("gates", []):
            for kp in g.get("keypoints_2d", {}).values():
                xy = kp.get("projected_px")
                if xy is not None:
                    x, y = (xy[0]-left)*sx, (xy[1]-top)*sy
                    kp["projected_px"] = [float(x), float(y)]
                    kp["normalized"] = [float(x/max(1,self.out_w-1)), float(y/max(1,self.out_h-1))]
                    kp["in_frame"] = bool(0 <= x < self.out_w and 0 <= y < self.out_h)
                    if kp.get("projection_valid", False) and not kp["in_frame"]:
                        kp["visible"] = False; kp["occluded"] = False; kp["visibility_state"] = "out_of_frame"
            bb = g.get("bounding_boxes", {})
            for key in ("visible_xyxy_px", "amodal_xyxy_px"):
                if bb.get(key) is not None:
                    x1,y1,x2,y2 = bb[key]
                    vals = [float((x1-left)*sx),float((y1-top)*sy),float((x2-left)*sx),float((y2-top)*sy)]
                    if key == "visible_xyxy_px":
                        vals = [max(0.0,min(float(self.out_w),vals[0])), max(0.0,min(float(self.out_h),vals[1])), max(0.0,min(float(self.out_w),vals[2])), max(0.0,min(float(self.out_h),vals[3]))]
                    bb[key] = vals
        frame["image"] = {"width_px": self.out_w, "height_px": self.out_h}
        return image, mask, frame

    def _photo(self, image):
        c = self.cfg
        if random.random() < float(c.get("brightness_p", 0.0)):
            image = ImageEnhance.Brightness(image).enhance(random.uniform(*c.get("brightness_range", [0.7,1.3])))
        if random.random() < float(c.get("contrast_p", 0.0)):
            image = ImageEnhance.Contrast(image).enhance(random.uniform(*c.get("contrast_range", [0.7,1.3])))
        if random.random() < float(c.get("color_p", 0.0)):
            image = ImageEnhance.Color(image).enhance(random.uniform(*c.get("color_range", [0.75,1.25])))
        if random.random() < float(c.get("blur_p", 0.0)):
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(*c.get("blur_radius", [0.2,1.5]))))
        if random.random() < float(c.get("jpeg_p", 0.0)):
            b = io.BytesIO(); q = random.randint(*c.get("jpeg_quality", [45,90])); image.save(b, "JPEG", quality=q); b.seek(0); image = Image.open(b).convert("RGB")
        if random.random() < float(c.get("noise_p", 0.0)):
            arr = np.asarray(image).astype(np.float32)
            sigma = random.uniform(*c.get("noise_sigma", [2.0,12.0]))
            arr = np.clip(arr + np.random.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)
            image = Image.fromarray(arr, "RGB")
        return image


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2,0,1).contiguous()
