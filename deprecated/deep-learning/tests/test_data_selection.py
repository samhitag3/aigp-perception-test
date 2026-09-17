from gateperceiver.data.dataset import _select_sequence_fraction


def test_seeded_sequence_fractions_are_deterministic_and_nested():
    seqs=[f"seq_{i:04d}" for i in range(100)]
    cheap=_select_sequence_fraction(seqs,0.15,42,"source_a")
    baseline=_select_sequence_fraction(seqs,0.50,42,"source_a")
    cheap2=_select_sequence_fraction(list(reversed(seqs)),0.15,42,"source_a")
    baseline2=_select_sequence_fraction(list(reversed(seqs)),0.50,42,"source_a")
    assert len(cheap)==15
    assert len(baseline)==50
    assert cheap==cheap2
    assert baseline==baseline2
    assert set(cheap).issubset(set(baseline))
