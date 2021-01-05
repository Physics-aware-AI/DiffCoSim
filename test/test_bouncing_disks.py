import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from datasets.datasets import RigidBodyDataset
from systems.bouncing_disks import BouncingDisks
from systems.chain_pendulum_with_wall import ChainPendulum_w_Wall
from pytorch_lightning import seed_everything

import torch

seed_everything(0)

import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")

def test_disks():
    body = BouncingDisks(n_disks=2, m=0.1, l=0.1, mu=0.1, cor=0.0)
    dataset = RigidBodyDataset(body=body, n_traj=5, chunk_len=100, regen=True)

    ani = body.animate(dataset.zs, 1)
    ani.save(os.path.join(THIS_DIR,'test_disks.gif'))

    assert 1


if __name__ == "__main__":
    test_disks()