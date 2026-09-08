from __future__ import annotations
import argparse,torch
from torch.utils.data import DataLoader
from gatepose.utils.config import load_config
from gatepose.data.canonical_dataset import CanonicalGateDataset,collate_canonical
from gatepose.models import build_model
from gatepose.evaluation.evaluator import evaluate
p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--dataset',required=True); p.add_argument('--split',choices=['train','validation','test'],default='test'); p.add_argument('--run-dir',required=True); p.add_argument('--device',default='cuda'); p.add_argument('--save-predictions',action='store_true'); a=p.parse_args()
cfg=load_config(a.config); dev=torch.device(a.device if torch.cuda.is_available() or a.device=='cpu' else 'cpu'); manifest={'train':'splits/train_sequences.txt','validation':'splits/validation_sequences.txt','test':'splits/test_sequences.txt'}[a.split]
ds=CanonicalGateDataset(a.dataset,manifest,int(cfg['training']['window_size']),(cfg['camera']['model_width'],cfg['camera']['model_height']),{"enabled":False}); dl=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0,collate_fn=collate_canonical)
model=build_model(cfg).to(dev); ck=torch.load(a.checkpoint,map_location=dev); model.load_state_dict(ck['model'] if 'model' in ck else ck); print(evaluate(model,dl,cfg,a.run_dir,dev,a.save_predictions))
