from __future__ import annotations
import argparse,copy
from pathlib import Path
import torch,optuna
from torch.utils.data import DataLoader
from gatepose.utils.config import load_config,save_config
from gatepose.utils.seed import seed_everything
from gatepose.data.canonical_dataset import CanonicalGateDataset,collate_canonical
from gatepose.models import build_model
from gatepose.training.engine import train
p=argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--dataset'); p.add_argument('--train-manifest'); p.add_argument('--val-manifest'); p.add_argument('--study-name',required=True); p.add_argument('--storage',required=True); p.add_argument('--run-dir',required=True); p.add_argument('--n-trials',type=int,default=25); p.add_argument('--device',default='cuda'); p.add_argument('--print-best',action='store_true'); a=p.parse_args()
study=optuna.create_study(study_name=a.study_name,storage=a.storage,direction='minimize',load_if_exists=True)
if a.print_best:
    print(study.best_trial.number,study.best_value,study.best_trial.params); raise SystemExit
base=load_config(a.config); dev=torch.device(a.device if torch.cuda.is_available() or a.device=='cpu' else 'cpu')
def objective(trial):
    cfg=copy.deepcopy(base); cfg['optimizer']['lr']=trial.suggest_float('lr',3e-5,1e-3,log=True); cfg['optimizer']['weight_decay']=trial.suggest_float('weight_decay',1e-6,1e-3,log=True); cfg['model']['decoder_layers']=trial.suggest_int('decoder_layers',1,3); cfg['loss']['w_keypoint']=trial.suggest_float('w_keypoint',2,8); cfg['loss']['w_position']=trial.suggest_float('w_position',2,10); cfg['loss']['w_rotation']=trial.suggest_float('w_rotation',1,5); cfg['loss']['blind_scale']=trial.suggest_float('blind_scale',.1,.6); cfg['model']['ego_dropout_probability']=trial.suggest_float('ego_dropout_probability',0,.4); cfg['training']['epochs']=int(base.get('optuna',{}).get('trial_epochs',25)); seed_everything(int(cfg.get('seed',42))+trial.number)
    dsargs=(a.dataset,int(cfg['training']['window_size']),(cfg['camera']['model_width'],cfg['camera']['model_height']))
    tr=CanonicalGateDataset(dsargs[0],a.train_manifest,dsargs[1],dsargs[2],cfg.get('augmentation',{})); va=CanonicalGateDataset(dsargs[0],a.val_manifest,dsargs[1],dsargs[2],{"enabled":False}); dl=lambda ds,sh:DataLoader(ds,batch_size=int(cfg['training']['batch_size']),shuffle=sh,num_workers=0,collate_fn=collate_canonical)
    rd=Path(a.run_dir)/f'trial_{trial.number:04d}'; model=build_model(cfg).to(dev); score=train(model,dl(tr,True),dl(va,False),cfg,rd,dev); save_config(cfg,rd/'config.yaml'); return score
study.optimize(objective,n_trials=a.n_trials)
best=copy.deepcopy(base)
for k,v in study.best_trial.params.items():
    if k in ('lr','weight_decay'): best['optimizer'][k]=v
    elif k=='decoder_layers': best['model'][k]=v
    elif k=='ego_dropout_probability': best['model'][k]=v
    elif k.startswith('w_') or k=='blind_scale': best['loss'][k]=v
Path(a.run_dir).mkdir(parents=True,exist_ok=True); save_config(best,Path(a.run_dir)/'best_config.yaml'); print('best',study.best_value,study.best_trial.params)
