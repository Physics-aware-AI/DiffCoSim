import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from datasets.datasets import RigidBodyDataset
from systems.bouncing_mass_points import BouncingMassPoints
from systems.chain_pendulum_with_wall import ChainPendulum_w_Wall
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer

import torch
import torch.nn as nn
import torch.nn.functional as F
from trainer import Model

seed_everything(0)

import matplotlib.pyplot as plt
# plt.switch_backend("TkAgg")

def test_infeasible():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BM3_homo_cor1_mu0_N800",
        "version_0",
        "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    print(torch.exp(model.model.m_params["0"]))
    print(F.hardsigmoid(model.model.cor_params))
    print(F.relu(model.model.mu_params))

    trainer = Trainer(resume_from_checkpoint=checkpoint_path)

    trainer.fit(model)

    assert 1


def test_unbounded():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BM2_homo_cor1_mu0_N800",
        "version_0",
        "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    print(torch.exp(model.model.m_params["0"]))
    print(F.hardsigmoid(model.model.cor_params))
    print(F.relu(model.model.mu_params))

    trainer = Trainer(resume_from_checkpoint=checkpoint_path)

    trainer.fit(model)

    assert 1

if __name__ == "__main__":
    test_unbounded()