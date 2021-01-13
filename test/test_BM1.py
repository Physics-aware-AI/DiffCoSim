import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from datasets.datasets import RigidBodyDataset
from systems.bouncing_mass_points import BouncingMassPoints
from systems.chain_pendulum_with_wall import ChainPendulum_w_Wall
from pytorch_lightning import seed_everything

import torch
import torch.nn as nn
import json
from trainer import Model

seed_everything(0)

import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")

def test_BM1_0():
    body_kwargs_file = "BM1_homo_cor1_mu0"
    with open(os.path.join(PARENT_DIR, "examples", body_kwargs_file + ".json")) as file:
        body_kwargs = json.load(file)
    body = BouncingMassPoints(body_kwargs_file, **body_kwargs)
    dataset = RigidBodyDataset(
        mode = "test",
        n_traj = 10,
        body = body,
        dtype = torch.float32,
        chunk_len = 100,
        regen=True
    )

    ani = body.animate(dataset.zs, 1)
    ani.save(os.path.join(THIS_DIR, 'test_BM1_0.gif'))

    assert 1

def test_BM1_1():
    body_kwargs_file = "BM1_homo_cor0_mu0"
    with open(os.path.join(PARENT_DIR, "examples", body_kwargs_file + ".json")) as file:
        body_kwargs = json.load(file)
    body = BouncingMassPoints(body_kwargs_file, **body_kwargs)
    dataset = RigidBodyDataset(
        mode = "test",
        n_traj = 10,
        body = body,
        dtype = torch.float32,
        chunk_len = 100,
        regen=False
    )

    ani = body.animate(dataset.zs, 2)
    ani.save(os.path.join(THIS_DIR, 'test_BM1_1.gif'))

    assert 1

def test_BM1_2():
    body_kwargs_file = "BM1_homo_cor1_mu0_g0"
    with open(os.path.join(PARENT_DIR, "examples", body_kwargs_file + ".json")) as file:
        body_kwargs = json.load(file)
    body = BouncingMassPoints(body_kwargs_file, **body_kwargs)
    dataset = RigidBodyDataset(
        mode = "test",
        n_traj = 10,
        body = body,
        dtype = torch.float32,
        chunk_len = 100,
        regen=True
    )

    ani = body.animate(dataset.zs, 2)
    ani.save(os.path.join(THIS_DIR, 'test_BM1_2.gif'))

    assert 1

if __name__ == "__main__":
    test_BM1_1()