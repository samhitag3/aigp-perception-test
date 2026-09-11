#!/usr/bin/env python3
"""Quick CUDA/GPU sanity check. Run on the remote server after `uv sync`.

    uv run python scripts/check_gpu.py
"""

import torch


def main():
    print(f"torch            : {torch.__version__}")
    print(f"cuda available   : {torch.cuda.is_available()}")
    print(f"cuda version     : {torch.version.cuda}")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"device count     : {n}")
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            print(f"  [{i}] {p.name}  {p.total_memory/1e9:.1f} GB  sm_{p.major}{p.minor}")
        # tiny matmul on GPU to confirm it actually works
        x = torch.randn(1024, 1024, device="cuda")
        y = (x @ x).sum().item()
        print(f"gpu matmul ok    : {y:.1f}")
    else:
        print("No CUDA device found — training will fall back to CPU.")


if __name__ == "__main__":
    main()
