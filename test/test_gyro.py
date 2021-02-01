import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from datasets.datasets import RigidBodyDataset
from systems.gyroscope_with_wall import GyroscopeWithWall
from pytorch_lightning import seed_everything

import torch
import torch.nn as nn
import json
from trainer import Model

seed_everything(0)

import matplotlib.pyplot as plt
# plt.switch_backend("TkAgg")

def test_gyro_0():
    body_kwargs_file = "Gyro_homo_cor1_mu0"
    with open(os.path.join(PARENT_DIR, "examples", body_kwargs_file + ".json")) as file:
        body_kwargs = json.load(file)
    body = GyroscopeWithWall(body_kwargs_file, **body_kwargs)
    dataset = RigidBodyDataset(
        mode = "test",
        n_traj = 10,
        body = body,
        dtype = torch.float32,
        chunk_len = 100,
        regen=False
    )

    # N, T = dataset.zs.shape[:2]
    # x, v = dataset.zs.chunk(2, dim=2)
    # p_x = body.M.type_as(v) @ v
    # zts = torch.cat([x, p_x], dim=2)
    # energy = body.hamiltonian(None, zts.reshape(N*T, -1)).reshape(N, T)
    # plt.plot(energy[3])
    # plt.show()

    ani = body.animate(dataset.zs[:, 0:2], 0)
    ani.save(os.path.join(THIS_DIR, 'test_gyro_0.gif'))

    assert 1

def test_gyro_1():
    body_kwargs_file = "Gyro_homo_cor0.8_mu0.1"
    with open(os.path.join(PARENT_DIR, "examples", body_kwargs_file + ".json")) as file:
        body_kwargs = json.load(file)
    body = GyroscopeWithWall(body_kwargs_file, **body_kwargs)
    dataset = RigidBodyDataset(
        mode = "test",
        n_traj = 10,
        body = body,
        dtype = torch.float32,
        chunk_len = 100,
        regen=False
    )

    N, T = dataset.zs.shape[:2]
    x, v = dataset.zs.chunk(2, dim=2)
    p_x = body.M.type_as(v) @ v
    zts = torch.cat([x, p_x], dim=2)
    energy = body.hamiltonian(None, zts.reshape(N*T, -1)).reshape(N, T)
    plt.plot(energy[2])
    plt.show()

    ani = body.animate(dataset.zs, 2)
    ani.save(os.path.join(THIS_DIR, 'test_gyro_1.gif'))

    assert 1

if __name__ == "__main__":
    test_gyro_0()