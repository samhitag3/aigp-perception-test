from __future__ import annotations
from collections import defaultdict, deque
import numpy as np
import torch
from PIL import Image

from gateperception.data.canonical import KEYPOINT_ORDER
from .decoder import Detection

VIS_ID_TO_STATE = {0: "invalid", 1: "visible", 2: "occluded", 3: "out_of_frame"}


def crop_rgba(rgb: np.ndarray, mask: np.ndarray, box: list[float], crop_size: tuple[int, int], padding: float) -> tuple[torch.Tensor, tuple[float, float, float, float]]:
    x0, y0, x1, y1 = map(float, box)
    bw, bh = max(2.0, x1 - x0), max(2.0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cw, ch = bw * (1 + 2 * padding), bh * (1 + 2 * padding)
    x0, y0, x1, y1 = cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2
    ix0, iy0, ix1, iy1 = int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1))
    w, h = max(2, ix1 - ix0), max(2, iy1 - iy0)
    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    sx0, sy0, sx1, sy1 = max(0, ix0), max(0, iy0), min(rgb.shape[1], ix1), min(rgb.shape[0], iy1)
    if sx1 > sx0 and sy1 > sy0:
        dx0, dy0 = sx0 - ix0, sy0 - iy0
        dx1, dy1 = dx0 + sx1 - sx0, dy0 + sy1 - sy0
        canvas[dy0:dy1, dx0:dx1, :3] = rgb[sy0:sy1, sx0:sx1]
        canvas[dy0:dy1, dx0:dx1, 3] = mask[sy0:sy1, sx0:sx1].astype(np.uint8) * 255
    im = Image.fromarray(canvas, mode="RGBA").resize(crop_size, Image.Resampling.BILINEAR)
    arr = np.asarray(im, dtype=np.uint8).copy()
    arr[..., 3] = (arr[..., 3] >= 128).astype(np.uint8) * 255
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0, (float(ix0), float(iy0), float(w), float(h))


class KeypointTrackMemory:
    def __init__(self, model, device: torch.device, window_size: int = 5, crop_size: tuple[int, int] = (192, 192), crop_padding: float = 0.35):
        self.model = model
        self.device = device
        self.window_size = int(window_size)
        self.crop_size = tuple(map(int, crop_size))
        self.crop_padding = float(crop_padding)
        self.histories: dict[str, deque[torch.Tensor]] = defaultdict(lambda: deque(maxlen=self.window_size))

    @torch.no_grad()
    def predict(self, rgb: np.ndarray, det: Detection) -> dict:
        assert det.track_id is not None
        crop, tr = crop_rgba(rgb, det.mask, det.box_xyxy_px, self.crop_size, self.crop_padding)
        hist = self.histories[det.track_id]
        hist.append(crop)
        frames = list(hist)
        while len(frames) < self.window_size:
            frames.insert(0, frames[0])
        x = torch.stack(frames, dim=0)[None].to(self.device)
        out = self.model(x)
        kp = out["keypoints"][0].detach().cpu().numpy()
        vis_prob = out["visibility_logits"][0].softmax(-1).detach().cpu().numpy()
        x0, y0, cw, ch = tr
        result = {}
        for i, name in enumerate(KEYPOINT_ORDER):
            px = [float(x0 + kp[i, 0] * cw), float(y0 + kp[i, 1] * ch)]
            vis_id = int(vis_prob[i].argmax())
            result[name] = {
                "xy_px": px,
                "confidence": float(1.0 - vis_prob[i, 0]),
                "visibility": VIS_ID_TO_STATE[vis_id],
                "visibility_probabilities": {
                    "invalid": float(vis_prob[i, 0]),
                    "visible": float(vis_prob[i, 1]),
                    "occluded": float(vis_prob[i, 2]),
                    "out_of_frame": float(vis_prob[i, 3]),
                },
            }
        return result
