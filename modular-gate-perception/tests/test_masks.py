import numpy as np


def test_instance_to_union_and_binary():
    m = np.array([[0,1,1],[2,0,2]], dtype=np.uint16)
    assert np.array_equal(m > 0, np.array([[False,True,True],[True,False,True]]))
    assert (m == 1).sum() == 2
    assert (m == 2).sum() == 2
