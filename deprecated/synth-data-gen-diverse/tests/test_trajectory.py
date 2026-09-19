import numpy as np
from gatesynth.config import load_config
from gatesynth.trajectory import generate_course, generate_camera_trajectory


def test_diverse_course_and_physical_motion():
    cfg = load_config('configs/smoke_real_camera_diverse.yaml')
    rng = np.random.default_rng(123)
    gates = generate_course(rng, cfg['course'])
    assert len(gates) >= cfg['course']['gates_min']
    headings = [g['course_heading_deg'] for g in gates]
    assert all(np.isfinite(headings))

    traj, meta = generate_camera_trajectory(rng, gates, cfg['trajectory'], cfg['motion'], cfg['sequences']['fps'])
    assert len(traj) > 2
    assert meta['speed_min_mps'] > 0
    assert meta['speed_max_mps'] >= meta['speed_min_mps']
    s = np.array([x['path_s_m'] for x in traj])
    assert np.all(np.diff(s) >= -1e-9)
    speeds = np.array([x['speed_mps'] for x in traj])
    assert np.all(np.isfinite(speeds))
    assert np.all(speeds > 0)
