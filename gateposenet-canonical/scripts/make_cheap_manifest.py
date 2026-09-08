from __future__ import annotations
import argparse,random
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--source',required=True); p.add_argument('--fraction',type=float,default=.15); p.add_argument('--seed',type=int,default=42); p.add_argument('--output',required=True); a=p.parse_args(); root=Path(a.dataset); src=Path(a.source); src=src if src.is_absolute() else root/src; out=Path(a.output); out=out if out.is_absolute() else root/out
ids=[x.strip() for x in src.read_text().splitlines() if x.strip()]; rng=random.Random(a.seed); rng.shuffle(ids); n=max(1,round(len(ids)*a.fraction)); chosen=sorted(ids[:n]); out.parent.mkdir(parents=True,exist_ok=True); out.write_text('\n'.join(chosen)+'\n'); print(f'wrote {n}/{len(ids)} sequences -> {out}')
