import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from datasets.datasets import RigidBodyDataset
from systems.bouncing_mass_points import BouncingMassPoints
from systems.chain_pendulum_with_contact import ChainPendulumWithContact
from pytorch_lightning import seed_everything

import torch
import torch.nn as nn
import json
from trainer import Model

seed_everything(0)

import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")

def test_CP3_0():
    body_kwargs_file = "CP3_ground_homo_cor1_mu0"
    with open(os.path.join(PARENT_DIR, "examples", body_kwargs_file + ".json")) as file:
        body_kwargs = json.load(file)
    body = ChainPendulumWithContact(body_kwargs_file, **body_kwargs)
    dataset = RigidBodyDataset(
        mode = "test",
        n_traj = 10,
        body = body,
        dtype = torch.float32,
        chunk_len = 100,
        regen=True,
        noise_std=0.05,
    )

    N, T = dataset.zs.shape[:2]
    x, v = dataset.zs.chunk(2, dim=2)
    p_x = body.M.type_as(v) @ v
    zts = torch.cat([x, p_x], dim=2)
    energy = body.hamiltonian(None, zts.reshape(N*T, -1)).reshape(N, T)
    plt.plot(energy[0])
    plt.show()

    ani = body.animate(dataset.zs, 0)
    ani.save(os.path.join(THIS_DIR, 'test_CP3_0.gif'))

    assert 1

if __name__ == "__main__":
    test_CP3_0()