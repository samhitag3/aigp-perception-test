from __future__ import annotations
import json,time
from pathlib import Path
import torch
from tqdm import tqdm
from gatepose.training.losses import compute_losses


def move_batch(batch,device):
    for k in ["images","intrinsics","ego"]: batch[k]=batch[k].to(device,non_blocking=True)
    return batch


def run_epoch(model,loader,optimizer,scaler,cfg,device,train=True):
    model.train(train); sums={}; n=0
    ctx=torch.enable_grad if train else torch.no_grad
    with ctx():
        for batch in tqdm(loader,leave=False):
            batch=move_batch(batch,device)
            if train: optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=(device.type=="cuda" and bool(cfg["training"].get("amp",True)))):
                out=model(batch["images"],batch["ego"],batch["intrinsics"]); loss,parts=compute_losses(out,batch,cfg)
            if train:
                scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg["training"].get("grad_clip_norm",1.0))); scaler.step(optimizer); scaler.update()
            n+=1
            for k,v in parts.items(): sums[k]=sums.get(k,0.0)+float(v.detach().cpu())
    return {k:v/max(n,1) for k,v in sums.items()}


def train(model,train_loader,val_loader,cfg,run_dir,device):
    run=Path(run_dir); run.mkdir(parents=True,exist_ok=True)
    opt=torch.optim.AdamW(model.parameters(),lr=float(cfg["optimizer"]["lr"]),weight_decay=float(cfg["optimizer"]["weight_decay"]))
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,int(cfg["training"]["epochs"])))
    scaler=torch.amp.GradScaler("cuda",enabled=(device.type=="cuda" and bool(cfg["training"].get("amp",True))))
    best=float("inf"); patience=0; hist=[]
    for epoch in range(1,int(cfg["training"]["epochs"])+1):
        t=time.time(); tr=run_epoch(model,train_loader,opt,scaler,cfg,device,True); va=run_epoch(model,val_loader,None,scaler,cfg,device,False); sched.step(); row={"epoch":epoch,"train":tr,"val":va,"seconds":time.time()-t,"lr":opt.param_groups[0]["lr"]}; hist.append(row); print(json.dumps(row))
        state={"model":model.state_dict(),"optimizer":opt.state_dict(),"epoch":epoch,"config":cfg,"val":va}; torch.save(state,run/"last.pt")
        score=va.get("total",float("inf"))
        if score<best: best=score; patience=0; torch.save(state,run/"best.pt")
        else: patience+=1
        (run/"training_history.json").write_text(json.dumps(hist,indent=2),encoding="utf-8")
        if patience>=int(cfg["training"].get("early_stopping_patience",999)): break
    return best
