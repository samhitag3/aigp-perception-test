from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from gatepose.utils.config import load_config,save_config
from gatepose.utils.seed import seed_everything
from gatepose.data.canonical_dataset import CanonicalGateDataset,collate_canonical
from gatepose.models import build_model
from gatepose.training.engine import train

p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--dataset',required=True); p.add_argument('--train-manifest',required=True); p.add_argument('--val-manifest',required=True); p.add_argument('--run-dir',required=True); p.add_argument('--device',default='cuda'); a=p.parse_args()
cfg=load_config(a.config); seed_everything(int(cfg.get('seed',42))); device=torch.device(a.device if torch.cuda.is_available() or a.device=='cpu' else 'cpu')
size=(cfg['camera']['model_width'],cfg['camera']['model_height']); ws=int(cfg['training']['window_size'])
tr=CanonicalGateDataset(a.dataset,a.train_manifest,ws,size,cfg.get('augmentation',{})); va=CanonicalGateDataset(a.dataset,a.val_manifest,ws,size,{"enabled":False})
dl=lambda ds,sh: DataLoader(ds,batch_size=int(cfg['training']['batch_size']),shuffle=sh,num_workers=int(cfg['training'].get('num_workers',4)),pin_memory=True,collate_fn=collate_canonical,persistent_workers=int(cfg['training'].get('num_workers',4))>0)
model=build_model(cfg).to(device); Path(a.run_dir).mkdir(parents=True,exist_ok=True); save_config(cfg,Path(a.run_dir)/'config.yaml')
print(f'device={device} train_windows={len(tr)} val_windows={len(va)} params={sum(p.numel() for p in model.parameters()):,}')
train(model,dl(tr,True),dl(va,False),cfg,a.run_dir,device)
