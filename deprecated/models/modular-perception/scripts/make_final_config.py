from __future__ import annotations
import argparse
from gateperception.utils.config import load_yaml, save_yaml, deep_update


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base-final",required=True)
    ap.add_argument("--tuned",required=True)
    ap.add_argument("--kind",choices=["segmentation","keypoints"],required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    base=load_yaml(args.base_final); tuned=load_yaml(args.tuned)
    if args.kind=="segmentation":
        patch={
          "training":{"learning_rate":tuned["training"]["learning_rate"],"weight_decay":tuned["training"]["weight_decay"]},
          "loss":{k:tuned["loss"][k] for k in ["mask_dice_weight","track_weight","object_pos_weight"]},
        }
    else:
        patch={
          "training":{"learning_rate":tuned["training"]["learning_rate"],"weight_decay":tuned["training"]["weight_decay"]},
          "model":{"crop_padding":tuned["model"]["crop_padding"]},
          "loss":{k:tuned["loss"][k] for k in ["coord_weight","visibility_weight"]},
        }
    save_yaml(deep_update(base,patch),args.output)
    print(args.output)

if __name__=="__main__": main()
