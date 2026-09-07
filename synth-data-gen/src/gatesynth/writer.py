from __future__ import annotations
from pathlib import Path
import hashlib, json
import cv2


def ensure_sequence_dirs(root: Path, seq_id: str, with_depth=False):
    seq=root/"sequences"/seq_id
    (seq/"rgb").mkdir(parents=True,exist_ok=True)
    (seq/"instance_masks").mkdir(parents=True,exist_ok=True)
    if with_depth:
        (seq/"depth_mm").mkdir(parents=True,exist_ok=True)
    return seq


def save_rgb(path: Path, rgb, quality=95):
    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path),bgr,[int(cv2.IMWRITE_JPEG_QUALITY),int(quality)])


def save_mask(path: Path, mask):
    cv2.imwrite(str(path),mask)


def json_dump(path: Path, obj):
    with open(path,"w",encoding="utf-8") as f:
        json.dump(obj,f,indent=2)


def write_jsonl(path: Path, records):
    with open(path,"w",encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r,separators=(",",":")) + "\n")


def write_checksums(root: Path):
    lines=[]
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name=="checksums.sha256":
            continue
        h=hashlib.sha256()
        with open(p,"rb") as f:
            for chunk in iter(lambda:f.read(1024*1024),b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {p.relative_to(root)}")
    (root/"checksums.sha256").write_text("\n".join(lines)+"\n",encoding="utf-8")
