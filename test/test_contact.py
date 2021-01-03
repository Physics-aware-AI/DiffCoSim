import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from datasets.datasets import RigidBodyDataset
from systems.bouncing_mass_points import BouncingMassPoints
from systems.chain_pendulum_with_wall import ChainPendulum_w_Wall
from pytorch_lightning import seed_everything

import torch

seed_everything(0)

import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")

def test_pendulum():
    body = BouncingMassPoints(n_balls=1, m=0.1, l=0.1, mu=0.0, cor=0.0)
    # initial condition 
    z0 = torch.zeros([1, 2, 1, 2], dtype=torch.float64)
    z0[0, 0, 0, 0] = 0.5 # x1
    z0[0, 0, 0, 1] = -0.2 # x2
    z0[0, 1, 0, 0] = 0.1 # v1
    z0[0, 1, 0, 1] = - z0[0, 0, 0, 1] / body.dt # v2

    ts = torch.arange(0, 5*body.dt, body.dt, device=z0.device, dtype=z0.dtype)
    zs = body.integrate(z0, ts)
    ani = body.animate(zs, 0)
    ani.save(os.path.join(THIS_DIR, "test_c.gif"))
    # make sure ball is bouncing back
    assert zs[0, 1, 0, 0, 1] > zs[0, 0, 0, 0, 1] 


if __name__ == "__main__":
    test_pendulum()