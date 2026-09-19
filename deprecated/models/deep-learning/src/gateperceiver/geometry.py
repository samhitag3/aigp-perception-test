from __future__ import annotations
import math
import torch


def fov_from_intrinsics(width: float, height: float, fx: float, fy: float) -> tuple[float, float]:
    hfov = math.degrees(2.0 * math.atan(width / (2.0 * fx)))
    vfov = math.degrees(2.0 * math.atan(height / (2.0 * fy)))
    return hfov, vfov


def box_xyxy_to_cxcywh(boxes: torch.Tensor, width: float, height: float) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([
        (x1 + x2) * 0.5 / width,
        (y1 + y2) * 0.5 / height,
        (x2 - x1).clamp_min(0) / width,
        (y2 - y1).clamp_min(0) / height,
    ], dim=-1)


def box_cxcywh_to_xyxy(boxes: torch.Tensor, width: float, height: float) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([
        (cx - 0.5 * w) * width,
        (cy - 0.5 * h) * height,
        (cx + 0.5 * w) * width,
        (cy + 0.5 * h) * height,
    ], dim=-1)


def box_iou_pairwise(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    a1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0))[:, None]
    a2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0))[None, :]
    union = a1 + a2 - inter
    return inter / union.clamp_min(1e-6)


def generalized_box_iou_cxcywh(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    # elementwise for same-length normalized cxcywh arrays
    def to_xyxy(b):
        cx, cy, w, h = b.unbind(-1)
        return torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], -1)
    a, b = to_xyxy(boxes1), to_xyxy(boxes2)
    lt = torch.maximum(a[:, :2], b[:, :2])
    rb = torch.minimum(a[:, 2:], b[:, 2:])
    wh = (rb-lt).clamp_min(0)
    inter = wh[:,0]*wh[:,1]
    area_a = ((a[:,2]-a[:,0]).clamp_min(0)*(a[:,3]-a[:,1]).clamp_min(0))
    area_b = ((b[:,2]-b[:,0]).clamp_min(0)*(b[:,3]-b[:,1]).clamp_min(0))
    union = area_a+area_b-inter
    iou = inter/union.clamp_min(1e-6)
    c_lt = torch.minimum(a[:,:2], b[:,:2]); c_rb = torch.maximum(a[:,2:], b[:,2:])
    c_area = ((c_rb[:,0]-c_lt[:,0]).clamp_min(0)*(c_rb[:,1]-c_lt[:,1]).clamp_min(0))
    return iou - (c_area-union)/c_area.clamp_min(1e-6)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1*a2).sum(-1, keepdim=True)*b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(R: torch.Tensor) -> torch.Tensor:
    return R[..., :2, :].reshape(*R.shape[:-2], 6)


def rotation_geodesic_deg(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    rel = R1 @ R2.transpose(-1, -2)
    tr = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))
