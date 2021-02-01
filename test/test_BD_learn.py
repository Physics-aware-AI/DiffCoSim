import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from datasets.datasets import RigidBodyDataset
from systems.bouncing_mass_points import BouncingMassPoints
from systems.bouncing_disks import BouncingDisks
from pytorch_lightning import seed_everything

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from trainer import Model

seed_everything(0)

import matplotlib.pyplot as plt
# plt.switch_backend("TkAgg")

def test_BD1_learn_0():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BD1_homo_cor0_mu0.5_N800",
        "version_0",
        "epoch=992.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    print(torch.exp(model.model.m_params["2"]))
    print(F.hardsigmoid(model.model.cor_params))
    print(F.relu(model.model.mu_params))
    # model.hparams.batch_size = 2
    # dataloader = model.test_dataloader()
    # test_batch = next(iter(dataloader))
    # log = model.test_step(test_batch, 0)
    # true_zts_true_energy = log["true_zts_true_energy"].numpy()
    # pred_zts_true_energy = log["pred_zts_true_energy"].numpy()
    # plt.plot(true_zts_true_energy[0])
    # plt.plot(pred_zts_true_energy[0])
    # plt.show()

    ani = model.body.animate(log["true_zts"], 0)
    ani.save(os.path.join(THIS_DIR, 'BD1_learn_0_true_zts.gif'))

    ani = model.body.animate(log["pred_zts"], 0)
    ani.save(os.path.join(THIS_DIR, "BD1_learn_0_pred_zts.gif"))

    assert 1

def test_BD5_learn_0():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BD5_hetero_g0_CLNNwC_N800",
        "version_2",
        "epoch=990.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    print(torch.exp(model.model.m_params["2"]))
    print(F.hardsigmoid(model.model.cor_params))
    print(F.relu(model.model.mu_params))
    model.hparams.batch_size = 2
    dataloader = model.test_dataloader()
    test_batch = next(iter(dataloader))
    log = model.test_step(test_batch, 0)

    ani = model.body.animate(log["true_zts"], 0)
    ani.save(os.path.join(THIS_DIR, 'BD5_learn_0_true_zts.gif'))

    ani = model.body.animate(log["pred_zts"], 0)
    ani.save(os.path.join(THIS_DIR, "BD5_learn_0_pred_zts.gif"))

if __name__ == "__main__":
    test_BD5_learn_0()