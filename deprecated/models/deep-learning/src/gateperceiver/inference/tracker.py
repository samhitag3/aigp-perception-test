from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment


def _cos(a,b):
    a=np.asarray(a); b=np.asarray(b); return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))

def _iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]); return inter/(aa+bb-inter+1e-8)

class OnlineTracker:
    def __init__(self,max_age=8,max_cost=0.8,embedding_weight=0.7):
        self.max_age=max_age; self.max_cost=max_cost; self.ew=embedding_weight; self.next_id=1; self.tracks={}
    def update(self, instances):
        ids=list(self.tracks); n,m=len(ids),len(instances)
        if n and m:
            C=np.zeros((n,m),np.float32)
            for i,tid in enumerate(ids):
                tr=self.tracks[tid]
                for j,ins in enumerate(instances): C[i,j]=self.ew*(1-_cos(tr["emb"],ins["track_embedding"]))+(1-self.ew)*(1-_iou(tr["box"],ins["box_xyxy_px"]))
            r,c=linear_sum_assignment(C); used=set()
            for i,j in zip(r,c):
                if C[i,j] <= self.max_cost:
                    tid=ids[i]; instances[j]["track_id"]=tid; self.tracks[tid]={"emb":instances[j]["track_embedding"],"box":instances[j]["box_xyxy_px"],"age":0}; used.add(j)
        else: used=set()
        for j,ins in enumerate(instances):
            if ins.get("track_id") is None:
                tid=f"track_{self.next_id:04d}"; self.next_id+=1; ins["track_id"]=tid; self.tracks[tid]={"emb":ins["track_embedding"],"box":ins["box_xyxy_px"],"age":0}
        alive={}
        active={ins["track_id"] for ins in instances}
        for tid,tr in self.tracks.items():
            if tid not in active: tr["age"]+=1
            if tr["age"]<=self.max_age: alive[tid]=tr
        self.tracks=alive
        for ins in instances: ins.pop("track_embedding",None)
        return instances
