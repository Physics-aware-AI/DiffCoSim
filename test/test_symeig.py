import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

import torch
from torch.autograd.gradcheck import get_analytical_jacobian
from symeig import symeig, symeig2

def test_original():
    n = 5
    a = torch.randn(n, n, dtype=torch.double, requires_grad=True)
    a0 = a + a.t()
    a1 = a0.clone()
    e0, v0 = torch.symeig(a0, eigenvectors=True)
    e1, v1 = symeig(a1)
    a2 = a0.clone()
    e2, v2 = symeig(a2)
    analytical0, *_ = get_analytical_jacobian((a0,),
                                                v0,
                                                nondet_tol=0.0)
    analytical1, *_ = get_analytical_jacobian((a1,),
                                                v1,
                                                nondet_tol=0.0)
    analytical2, *_ = get_analytical_jacobian((a2,),
                                                v2,
                                                nondet_tol=0.0)

    assert torch.allclose(analytical0[0], analytical1[0])
    assert torch.allclose(analytical0[0], analytical2[0])
    # e2, v2 = symeig2(a2)

if __name__ == "__main__":
    test_original()