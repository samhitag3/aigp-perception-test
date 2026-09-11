"""gatetest — standardized edge-case scorecards for gate-pose models.

Drop dataset directories (sam3-autolabeler `ground_truth.json` format) into
``data/test_drops/`` and run ``scripts/run_gate_tests.py``: every dropped
dataset is streamed through the model and scored on the STANDARD SLICES
(maneuver kind, visibility/blind, occlusion level, attitude, range, gates in
scene, target switches), producing scorecard.json + scorecard.md.
"""

from .scorecard import build_scorecard, format_scorecard_md  # noqa: F401
