import numpy as np

from gateposenet.mask_noise import WindowMaskNoise


def _sample_ids():
    x = np.zeros((180, 320), dtype=np.uint16)
    # Two square rings with different IDs.
    x[30:110, 30:38] = 1; x[30:110, 100:108] = 1
    x[30:38, 30:108] = 1; x[102:110, 30:108] = 1
    x[55:160, 170:178] = 2; x[55:160, 275:283] = 2
    x[55:63, 170:283] = 2; x[152:160, 170:283] = 2
    return x


def test_disabled_noise_is_exact_clean_union():
    ids = _sample_ids()
    n = WindowMaskNoise({"enabled": False}, seed=1)
    got = n.apply(ids, 0)
    want = (ids > 0).astype(np.uint8) * 255
    assert np.array_equal(got, want)


def test_noise_is_binary_and_does_not_modify_source_ids():
    ids = _sample_ids()
    before = ids.copy()
    cfg = {
        "enabled": True,
        "clean_window_prob": 0.0,
        "severity_min": 1.0,
        "severity_max": 1.0,
        "rounded_edge_prob": 1.0,
        "morph_prob": 1.0,
        "incomplete_prob": 1.0,
        "max_missing_chunks": 3,
        "gate_dropout_prob": 0.0,
        "full_frame_dropout_prob": 0.0,
        "persistent_false_artifact_prob": 1.0,
        "max_persistent_false_artifacts": 2,
        "small_false_blob_prob": 1.0,
        "max_small_false_blobs": 3,
    }
    n = WindowMaskNoise(cfg, seed=7)
    got = n.apply(ids, 0)
    assert np.array_equal(ids, before)
    assert set(np.unique(got).tolist()).issubset({0, 255})
    clean = (ids > 0).astype(np.uint8) * 255
    assert not np.array_equal(got, clean)


def test_same_seed_reproduces_window_noise():
    ids = _sample_ids()
    cfg = {"enabled": True, "clean_window_prob": 0.0,
           "severity_min": 0.8, "severity_max": 0.8}
    a = WindowMaskNoise(cfg, seed=123)
    b = WindowMaskNoise(cfg, seed=123)
    for t in range(4):
        assert np.array_equal(a.apply(ids, t), b.apply(ids, t))
