import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from datasets.datasets import RigidBodyDataset
from systems.rope_chain import RopeChain
from systems.bouncing_disks import BouncingDisks
from systems.bouncing_mass_points import BouncingMassPoints
from systems.elastic_rope import ElasticRope
from systems.gyroscope_with_wall import GyroscopeWithWall
from pytorch_lightning import seed_everything

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from trainer import Model

seed_everything(0)

import matplotlib.pyplot as plt
# plt.switch_backend("TkAgg")

def utils(model, name, n_p_1=0, test_batch_size=2, test_batch_idx=0):
    print(torch.exp(model.model.m_params[f"{n_p_1}"]))
    print("cor:" + f"{F.hardsigmoid(model.model.cor_params)}")
    print("mu:", f"{F.relu(model.model.mu_params)}")
    model.hparams.batch_size = test_batch_size
    dataloader = model.test_dataloader()
    test_batch = next(iter(dataloader))
    log = model.test_step(test_batch, 0)
    true_zts_true_energy = log["true_zts_true_energy"].numpy()
    pred_zts_true_energy = log["pred_zts_true_energy"].numpy()
    plt.plot(true_zts_true_energy[test_batch_idx])
    plt.plot(pred_zts_true_energy[test_batch_idx])
    plt.show(block=False)

    ani = model.body.animate(log["true_zts"], test_batch_idx)
    ani.save(os.path.join(THIS_DIR, f'{name}_true_zts.gif'))

    ani = model.body.animate(log["pred_zts"], test_batch_idx)
    ani.save(os.path.join(THIS_DIR, f"{name}_pred_zts.gif"))

def test_paper_0():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BM1_homo_cor1_mu0_CLNNwC_N800",
        "version_0",
        "epoch=922.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, name="test_paper_0")
    assert 1

def test_paper_1():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BM5_homo_cor1_mu0_g0_CLNNwC_N800",
        "version_0",
        "epoch=892.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, name="test_paper_1")
    assert 1

def test_paper_2():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BM5_hetero_g0_CLNNwC_N800",
        "version_0",
        "epoch=956.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, name="test_paper_2")
    assert 1

def test_paper_3():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BD1_homo_cor0_mu0.5_CLNNwC_N800",
        "version_0",
        "epoch=992.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=2, name="test_paper_3")
    assert 1

def test_paper_4():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BD5_hetero_g0_CLNNwC_N800",
        "version_2",
        "epoch=990.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=2, name="test_paper_4")
    assert 1

def test_paper_5():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "CP3_ground_homo_cor1_mu0_CLNNwC_N800",
        "version_0",
        "epoch=939.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0, name="test_paper_5")
    assert 1

def test_paper_6():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "CP3_ground_homo_cor0_mu0.5_CLNNwC_N800",
        "version_0",
        "epoch=969.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0, name="test_paper_6")
    assert 1

def test_paper_7():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "RC_default_CLNNwC_N800",
        "version_1", # 0
        "epoch=127.ckpt" # 903
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0, name="test_paper_7")
    assert 1

def test_paper_8():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER_default_CLNNwC_N800",
        "version_0",
        "epoch=979.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0, name="test_paper_8")
    assert 1

def test_paper_9():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "Gyro_homo_cor1_mu0_mujoco_CLNNwC_N800",
        "version_0",
        "epoch=856.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=3, name="test_paper_9")
    assert 1

def test_paper_10():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "Gyro_homo_cor0.8_mu0.1_mujoco_CLNNwC_N800",
        "version_0",
        "epoch=894.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=3, name="test_paper_10")
    assert 1

if __name__ == "__main__":
    test_paper_10()
