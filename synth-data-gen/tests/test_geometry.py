import numpy as np
from gatesynth.geometry import gate_keypoints_local, OUTER_W, INNER_W

def test_gate_dimensions():
    k=gate_keypoints_local()
    assert np.isclose(k["outer_tr"][0]-k["outer_tl"][0],OUTER_W)
    assert np.isclose(k["inner_tr"][0]-k["inner_tl"][0],INNER_W)
