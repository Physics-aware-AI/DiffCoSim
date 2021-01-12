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
from trainer import Model

seed_everything(0)

import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")

def test_teststep():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BM3_homo_cor1_mu0_N800",
        "version_0",
        "epoch=753.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    model.hparams.batch_size = 2
    dataloader = model.test_dataloader()
    test_batch = next(iter(dataloader))
    log = model.test_step(test_batch, 0)
    true_zts_true_energy = log["true_zts_true_energy"].numpy()
    pred_zts_true_energy = log["pred_zts_true_energy"].numpy()
    # plt.plot(true_zts_true_energy[0])
    # plt.plot(pred_zts_true_energy[0])
    # plt.show()

    ani = model.body.animate(log["true_zts"], 0)
    ani.save(os.path.join(THIS_DIR, 'test_teststep_true_zts.gif'))

    ani = model.body.animate(log["pred_zts"], 0)
    ani.save(os.path.join(THIS_DIR, "test_teststep_pred_zts.gif"))

    assert 1

def test_teststep_BM1():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BM1_homo_cor1_mu0_N800",
        "version_0",
        "epoch=848.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    # model.model.cor_params = nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32))
    model.hparams.batch_size = 2
    dataloader = model.test_dataloader()
    test_batch = next(iter(dataloader))
    log = model.test_step(test_batch, 0)
    true_zts_true_energy = log["true_zts_true_energy"].numpy()
    pred_zts_true_energy = log["pred_zts_true_energy"].numpy()
    # plt.plot(true_zts_true_energy[0])
    # plt.plot(pred_zts_true_energy[0])
    # plt.show()

    ani = model.body.animate(log["true_zts"], 0)
    ani.save(os.path.join(THIS_DIR, 'test_teststep_BM1_true_zts.gif'))

    ani = model.body.animate(log["pred_zts"], 0)
    ani.save(os.path.join(THIS_DIR, "test_teststep_BM1_pred_zts.gif"))

    assert 1


if __name__ == "__main__":
    test_teststep_BM1()