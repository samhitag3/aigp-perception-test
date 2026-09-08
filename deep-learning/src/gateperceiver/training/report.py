from __future__ import annotations
from datetime import datetime, timezone


def build_evaluation_report(cfg: dict, dataset_meta: dict, metrics: dict, model, score: float, split: str="validation") -> dict:
    params=sum(p.numel() for p in model.parameters()); trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
      "schema_version":"1.0.0",
      "run":{"created_at":datetime.now(timezone.utc).isoformat(),"model_name":"GatePerceiver","task_type":"multi_task_instance_segmentation_keypoints_pose_tracking","random_seed":cfg.get("seed")},
      "model":{"architecture":"GatePerceiver","backbone":cfg["model"].get("backbone"),"temporal":cfg["data"].get("window_size",1)>1,"temporal_window_size":cfg["data"].get("window_size",1),"input":{"modalities":["rgb","camera_intrinsics","gate_geometry"],"image_width":cfg["data"]["image_size"][1],"image_height":cfg["data"]["image_size"][0],"uses_camera_intrinsics":True,"uses_gate_geometry":True,"uses_previous_frames":cfg["data"].get("window_size",1)>1},"output":{"segmentation":True,"segmentation_type":"instance","keypoints":True,"pose":True,"gate_count":True,"visibility":True,"tracking_embeddings":True},"complexity":{"parameter_count":params,"trainable_parameter_count":trainable,"model_size_mb":None,"flops":None,"macs":None}},
      "dataset":{"dataset_name":dataset_meta.get("dataset_id"),"dataset_version":dataset_meta.get("dataset_id"),"schema_version":dataset_meta.get("schema_version"),"split_strategy":"sequence"},
      "evaluation":{"evaluation_split":split,
        "segmentation":{"available":True,"pixel_metrics":{"iou_mean":metrics.get("mean_instance_iou"),"dice_mean":metrics.get("mean_dice"),"precision":metrics.get("pixel_precision"),"recall":metrics.get("pixel_recall"),"f1":None,"false_positive_rate":None,"false_negative_rate":None},"instance_metrics":{"mean_instance_iou":metrics.get("mean_instance_iou"),"gate_instance_precision":metrics.get("gate_precision"),"gate_instance_recall":metrics.get("gate_recall"),"gate_instance_f1":metrics.get("gate_f1"),"missed_gate_rate":metrics.get("missed_gate_rate")}},
        "keypoints":{"available":True,"all_keypoints":{"mean_pixel_error":metrics.get("mean_keypoint_error_px"),"median_pixel_error":None,"mean_normalized_error":None,"pck":{"threshold_0.01":None,"threshold_0.02":None,"threshold_0.05":None,"threshold_0.10":None}},"visible_keypoints":{},"occluded_keypoints":{},"visibility_classification":{}},
        "pose":{"available":True,"translation":{"mean_error_m":metrics.get("mean_translation_error_m"),"median_error_m":None,"p90_error_m":None,"p95_error_m":None,"p99_error_m":None},"rotation":{"mean_error_deg":metrics.get("mean_rotation_error_deg"),"median_error_deg":None,"p90_error_deg":None,"p95_error_deg":None,"p99_error_deg":None},"position_components":{},"orientation_components":{},"geometry":{}},
        "gate_detection":{"gate_detection_rate":metrics.get("gate_recall"),"precision":metrics.get("gate_precision"),"recall":metrics.get("gate_recall"),"f1":metrics.get("gate_f1"),"false_positive_gates_per_frame":metrics.get("false_positive_gates_per_frame"),"catastrophic_failure_rate":metrics.get("missed_gate_rate")},
        "temporal":{"available":cfg["data"].get("window_size",1)>1,"tracking_accuracy":None,"id_switches":None,"track_fragmentation":None,"keypoint_jitter_px":None,"pose_jitter_translation_m":None,"pose_jitter_rotation_deg":None},
        "runtime":{"device":None,"latency_ms":{"mean":None,"median":None,"p90":None,"p95":None,"p99":None},"fps":{"mean":None}},
        "robustness":{"by_gate_distance":{},"by_gate_pixel_size":{},"by_occlusion":{},"by_frame_truncation":{},"by_gate_count":{},"by_view_angle":{},"by_motion":{},"by_image_condition":{},"by_domain":{}},
        "closed_loop":{"available":False,"gate_traversal_success_rate":None}
      },
      "leaderboard":{"primary_metric":"composite_perception_score","primary_metric_value":float(score),"secondary_metrics":metrics,"composite_score":float(score),"constraints":{},"passes_constraints":None},
      "notes":{"known_failure_modes":[],"observations":[],"recommended_next_steps":[]}
    }
