from __future__ import annotations
from pathlib import Path
import csv, json, time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from gateperceiver.data import GateSequenceDataset, collate_gate_batch
from gateperceiver.models import GatePerceiver
from gateperceiver.training.losses import compute_losses
from gateperceiver.training.metrics import matched_batch_stats, reduce_stats
from gateperceiver.training.per_sample import frame_row
from gateperceiver.training.report import build_evaluation_report
from gateperceiver.utils.seed import seed_everything, seed_worker
from gateperceiver.utils.io import write_json


def move_targets(targets, device):
    out=[]
    for seq in targets:
        row=[]
        for t in seq:
            row.append({k:(v.to(device) if torch.is_tensor(v) else v) for k,v in t.items()})
        out.append(row)
    return out


def build_loader(cfg, split, shuffle=False):
    d=cfg["data"]
    roots=d.get("roots") or [d.get("root")]
    roots=[r for r in roots if r]
    if not roots:
        raise ValueError("No dataset root configured; set data.root/data.roots or pass --dataset-root")
    max_samples=d.get(f"max_{split}_samples")
    fraction=float(d.get(f"{split}_sequence_fraction", 1.0))
    seed=int(cfg.get("seed",42))
    ds=GateSequenceDataset(
        roots, split, window_size=d.get("window_size",1), stride=d.get("stride",1),
        image_size=tuple(d.get("image_size",[360,640])),
        augment=(d.get("augmentation",{}) if split=="train" else {}),
        max_samples=max_samples, sequence_fraction=fraction, seed=seed,
    )
    generator=torch.Generator()
    generator.manual_seed(seed + {"train":0,"validation":1000,"test":2000}.get(split,3000))
    loader=DataLoader(
        ds, batch_size=int(cfg["training"].get("batch_size",2)), shuffle=shuffle,
        num_workers=int(d.get("num_workers",4)), pin_memory=True,
        collate_fn=collate_gate_batch,
        drop_last=(shuffle and len(ds)>=int(cfg["training"].get("batch_size",2))),
        worker_init_fn=seed_worker, generator=generator,
    )
    return ds,loader


def build_optimizer(model,cfg):
    t=cfg["training"]; name=t.get("optimizer","adamw").lower(); lr=float(t.get("lr",2e-4)); wd=float(t.get("weight_decay",1e-4))
    return torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd) if name=="adamw" else torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)


def train_experiment(cfg: dict, run_dir: str|Path, device="cuda", trial=None):
    run=Path(run_dir); run.mkdir(parents=True,exist_ok=True); seed_everything(int(cfg.get("seed",42)), deterministic=bool(cfg.get("deterministic", True)))
    dev=torch.device(device if device!="cuda" or torch.cuda.is_available() else "cpu")
    train_ds,train_loader=build_loader(cfg,"train",True); val_ds,val_loader=build_loader(cfg,"validation",False)
    selection = {
        "seed": int(cfg.get("seed",42)),
        "train_sequence_fraction": float(cfg.get("data",{}).get("train_sequence_fraction",1.0)),
        "validation_sequence_fraction": float(cfg.get("data",{}).get("validation_sequence_fraction",1.0)),
        "train": {"selected_sequences": train_ds.selected_sequences, "num_windows": len(train_ds)},
        "validation": {"selected_sequences": val_ds.selected_sequences, "num_windows": len(val_ds)},
    }
    write_json(selection, run/"data_selection.json")
    print("data roots:", [str(x) for x in train_ds.roots])
    print("train sequence fraction:", selection["train_sequence_fraction"], "seed:", selection["seed"])
    print("selected train sequences:", {k: len(v) for k,v in train_ds.selected_sequences.items()}, "windows:", len(train_ds))
    print("validation sequences:", {k: len(v) for k,v in val_ds.selected_sequences.items()}, "windows:", len(val_ds))
    model=GatePerceiver(cfg).to(dev); opt=build_optimizer(model,cfg); epochs=int(cfg["training"].get("epochs",20)); amp=bool(cfg["training"].get("amp",True) and dev.type=="cuda")
    scaler=torch.amp.GradScaler("cuda",enabled=amp); hist=[]; best=-1e9; best_metrics={}
    for epoch in range(1,epochs+1):
        model.train(); losses=[]; pbar=tqdm(train_loader,desc=f"epoch {epoch}/{epochs}")
        for batch in pbar:
            rgb=batch["rgb"].to(dev,non_blocking=True); K=batch["intrinsics"].to(dev,non_blocking=True); geom=batch["gate_geometry"].to(dev,non_blocking=True); tar=move_targets(batch["targets"],dev); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=amp):
                out=model(rgb,K,geom); parts,_=compute_losses(out,tar,cfg); loss=parts["total"]
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg["training"].get("grad_clip",1.0))); scaler.step(opt); scaler.update(); losses.append(float(loss.detach())); pbar.set_postfix(loss=f"{losses[-1]:.3f}")
        val,val_rows=evaluate_model(model,val_loader,cfg,dev,return_rows=True); score=selection_score(val,cfg); row={"epoch":epoch,"train_loss":sum(losses)/max(1,len(losses)),"selection_score":score,**val}; hist.append(row)
        torch.save({"model":model.state_dict(),"config":cfg,"epoch":epoch,"metrics":val},run/"last.pt")
        if score>best:
            best=score; best_metrics=val; torch.save({"model":model.state_dict(),"config":cfg,"epoch":epoch,"metrics":val},run/"best.pt")
        _write_history(hist,run/"training_history.csv"); _write_history(val_rows,run/"per_sample_metrics.csv")
        report=build_evaluation_report(cfg,train_ds.dataset_meta,best_metrics,model,best,"validation"); write_json(report,run/"evaluation.json")
        if trial is not None:
            trial.report(score,epoch)
            if trial.should_prune():
                import optuna; raise optuna.TrialPruned()
    return best,best_metrics


def selection_score(m,cfg):
    # tuned for safety-oriented perception; all terms higher is better except errors
    return float(0.30*m.get("gate_recall",0)+0.25*m.get("mean_instance_iou",0)+0.15*m.get("pixel_precision",0)+0.10*m.get("pixel_recall",0)+0.10*torch.exp(torch.tensor(-m.get("mean_translation_error_m",99.)/0.5)).item()+0.10*torch.exp(torch.tensor(-m.get("mean_rotation_error_deg",99.)/15.)).item())

@torch.no_grad()
def evaluate_model(model,loader,cfg,device,return_rows=False):
    model.eval(); stats=[]; rows=[]
    threshold=float(cfg.get("decode",{}).get("detection_threshold",0.5))
    for batch in tqdm(loader,desc="val",leave=False):
        rgb=batch["rgb"].to(device); K=batch["intrinsics"].to(device); geom=batch["gate_geometry"].to(device); tar=move_targets(batch["targets"],device); out=model(rgb,K,geom); _,matches=compute_losses(out,tar,cfg); stats.append(matched_batch_stats(out,tar,matches,threshold))
        if return_rows:
            B,T=out["pred_logits"].shape[:2]
            for b in range(B):
                for t in range(T): rows.append(frame_row(out,tar,matches,b,t,batch["meta"],threshold))
    metrics=reduce_stats(stats)
    return (metrics,rows) if return_rows else metrics


def _write_history(rows,path):
    if not rows:return
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
