"""GatePoseNet — temporal multimodal gate-relative pose regression.

A lightweight rolling-window perception model that consumes the FULL analytic
ground truth of the sam3-autolabeler synthetic trajectory datasets
(ground_truth.json): per-frame exact corners, center, 6-DoF camera-relative
gate pose, visibility, ego-motion and per-sequence intrinsics.

See gateposenet/README.md for the architecture and design rationale.
"""

from .model import (GatePoseNet, GatePoseNetMG,  # noqa: F401
                    build_gateposenet, build_gateposenet_mg)
from .model_single import (GatePoseNetSingle,  # noqa: F401
                           build_gateposenet_single)
from .dataset import GateSequenceDataset  # noqa: F401
from .losses import GatePoseLoss  # noqa: F401
from .canonical_dataset import CanonicalGateSequenceDataset  # noqa: F401
