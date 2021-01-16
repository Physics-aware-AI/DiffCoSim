import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from datasets.datasets import RigidBodyDataset
from systems.bouncing_mass_points import BouncingMassPoints
from pytorch_lightning import seed_everything

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from trainer import Model

seed_everything(0)

import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")

def test_BM2_learn_0():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BM2_homo_cor1_mu0_N800",
        "version_0",
        "epoch=976.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    print(torch.exp(model.model.m_params["0"]))
    print(F.hardsigmoid(model.model.cor_params))
    print(F.relu(model.model.mu_params))
    model.hparams.batch_size = 2
    dataloader = model.test_dataloader()
    test_batch = next(iter(dataloader))
    log = model.test_step(test_batch, 0)
    true_zts_true_energy = log["true_zts_true_energy"].numpy()
    pred_zts_true_energy = log["pred_zts_true_energy"].numpy()
    plt.plot(true_zts_true_energy[0])
    plt.plot(pred_zts_true_energy[0])
    plt.show()

    ani = model.body.animate(log["true_zts"], 0)
    ani.save(os.path.join(THIS_DIR, 'BM2_learn_0_true_zts.gif'))

    ani = model.body.animate(log["pred_zts"], 0)
    ani.save(os.path.join(THIS_DIR, "BM2_learn_0_pred_zts.gif"))

    assert 1

def test_BM2_learn_1():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BM2_homo_cor0.5_mu0.5_N800",
        "version_0",
        "epoch=818.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    print(torch.exp(model.model.m_params["0"]))
    print(F.hardsigmoid(model.model.cor_params))
    print(F.relu(model.model.mu_params))
    model.hparams.batch_size = 2
    dataloader = model.test_dataloader()
    test_batch = next(iter(dataloader))
    log = model.test_step(test_batch, 0)
    true_zts_true_energy = log["true_zts_true_energy"].numpy()
    pred_zts_true_energy = log["pred_zts_true_energy"].numpy()
    plt.plot(true_zts_true_energy[0])
    plt.plot(pred_zts_true_energy[0])
    plt.show()

    ani = model.body.animate(log["true_zts"], 0)
    ani.save(os.path.join(THIS_DIR, 'BM2_learn_1_true_zts.gif'))

    ani = model.body.animate(log["pred_zts"], 0)
    ani.save(os.path.join(THIS_DIR, "BM2_learn_1_pred_zts.gif"))

    assert 1

if __name__ == "__main__":
    test_BM2_learn_1()