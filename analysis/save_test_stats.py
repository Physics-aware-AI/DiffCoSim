"""
Non-commercial Use License

Copyright (c) 2021 Siemens Technology

This software, along with associated documentation files (the "Software"), is 
provided for the sole purpose of providing Proof of Concept. Any commercial 
uses of the Software including, but not limited to, the rights to sublicense, 
and/or sell copies of the Software are prohibited and are subject to a 
separate licensing agreement with Siemens. This software may be proprietary 
to Siemens and may be covered by patent and copyright laws. Processes 
controlled by the Software are patent pending.

The above copyright notice and this permission notice shall remain attached 
to the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)
import pickle
from argparse import ArgumentParser

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


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def utils(model, n_p_1=0):
    # IP = "IPR" if hasattr(model.hparams, "is_mujoco_like") and model.hparams.is_mujoco_like else "IP"
    # name = model.hparams.network_class + "_" + IP + "_" + model.hparams.body_kwargs_file
    # name = model.hparams.network_class + "_" + model.hparams.body_kwargs_file
    # name = model.hparams.network_class + "_lcp_" + model.hparams.body_kwargs_file
    # name = model.hparams.network_class + "_large_" + model.hparams.body_kwargs_file
    name = model.hparams.network_class + "_lcp_val_" + model.hparams.body_kwargs_file
    print(name)
    # print(torch.exp(model.model.m_params[f"{n_p_1}"]))
    # print(model.model.cor_params)
    # print("cor:" + f"{F.hardsigmoid(model.model.cor_params)}")
    # print("mu:", f"{F.relu(model.model.mu_params)}")
    model.hparams.batch_size = 200
    # test_dataset = str_to_class(model.hparams.dataset_class)(
    #     mode = "test",
    #     n_traj = 100,
    #     body = model.body,
    #     dtype = model.dtype,
    #     chunk_len = 100,
    #     noise_std = 0,
    # )
    # model.dataset["test"] = test_dataset
    model.eval()
    dataloader = model.test_dataloader()
    test_batch = next(iter(dataloader))
    with torch.no_grad():
        log = model.test_step(test_batch, 0, integration_time=6 * 0.005)
        # log = model.test_step(test_batch, 0)
    log["name"] = name
    # log["cor"] = F.hardsigmoid(model.model.cor_params)
    # log["mu"] = F.relu(model.model.mu_params)

    with open(os.path.join(THIS_DIR, name), "wb") as handle:
        pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)

#############
# CLNNwC
def save_paper_0():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BM5_homo_cor1_mu0_g0_CLNNwC_N800",
        "version_0",
        "epoch=892.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)

def save_paper_1():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BM5_hetero_g0_CLNNwC_N800",
        "version_0",
        "epoch=956.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)

def save_paper_2():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BD5_hetero_g0_CLNNwC_N800",
        "version_2",
        "epoch=990.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=2)
    assert 1

def save_paper_3():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "CP3_ground_homo_cor1_mu0_CLNNwC_N800",
        "version_0",
        "epoch=939.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

def save_paper_4():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "CP3_ground_homo_cor0_mu0.5_CLNNwC_N800",
        "version_0",
        "epoch=969.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

def save_paper_5():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER200k50_CLNNwC_N800",
        "version_0",
        "epoch=13.ckpt"
        # "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

def save_paper_6():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "Gyro_homo_cor1_mu0_CLNNwC_N800",
        "version_1",
        "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=3)
    assert 1

def save_paper_7():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "Gyro_homo_cor0.8_mu0.1_CLNNwC_N800",
        "version_1",
        "epoch=906.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=3)
    assert 1

def save_paper_8():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "Cloth_8K200_CLNNwC_N800",
        "version_0",
        # "epoch=13.ckpt"
        "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)
    assert 1

def save_paper_9():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER200k50_CLNNwC_N800",
        "version_2",
        # "epoch=13.ckpt"
        "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

def save_paper_91():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER_default_CLNNwC_N800",
        "version_3",
        # "epoch=13.ckpt"
        "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1


#########################################
# mujoco + CLNNwC
def save_paper_10():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BM5_homo_cor1_mu0_g0_mujoco_CLNNwC_N800",
        "version_0",
        "epoch=997.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)

def save_paper_11():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BM5_hetero_g0_mujoco_CLNNwC_N800",
        "version_0",
        "epoch=870.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)

def save_paper_12():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BD5_hetero_g0_mujoco_CLNNwC_N800",
        "version_0",
        "epoch=990.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=2)
    assert 1

def save_paper_13():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "CP3_ground_homo_cor1_mu0_mujoco_CLNNwC_N800",
        "version_0",
        "epoch=994.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

def save_paper_14():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "CP3_ground_homo_cor0_mu0.5_mujoco_CLNNwC_N800",
        "version_0",
        "epoch=979.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

def save_paper_15():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER200k50_mujoco_CLNNwC_N800",
        "version_0",
        "epoch=9.ckpt"
        # "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

def save_paper_16():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "Gyro_homo_cor1_mu0_mujoco_CLNNwC_N800",
        "version_0",
        "epoch=856.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=3)
    assert 1

def save_paper_17():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "Gyro_homo_cor0.8_mu0.1_mujoco_CLNNwC_N800",
        "version_0",
        "epoch=894.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=3)
    assert 1

def save_paper_18():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "Cloth_8K200_mujoco_CLNNwC_N800",
        "version_0",
        # "epoch=13.ckpt"
        "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)
    assert 1

def save_paper_19():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER200k50_mujoco_CLNNwC_N800",
        "version_2",
        # "epoch=9.ckpt"
        "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

def save_paper_191():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER_default_mujoco_CLNNwC_N800",
        "version_2",
        # "epoch=9.ckpt"
        "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1


###############3
# CHNNwC

def save_paper_20():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BM5_homo_cor1_mu0_g0_CHNNwC_N800",
        "version_0",
        "epoch=909.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)

def save_paper_21():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BM5_hetero_g0_CHNNwC_N800",
        "version_0",
        "epoch=956.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)

def save_paper_22():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BD5_hetero_g0_CHNNwC_N800",
        "version_0",
        "epoch=990.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=2)
    assert 1

def save_paper_23():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "CP3_ground_homo_cor1_mu0_CHNNwC_N800",
        "version_1",
        "epoch=950.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

def save_paper_24():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "CP3_ground_homo_cor0_mu0.5_CHNNwC_N800",
        "version_0",
        "epoch=983.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

def save_paper_25():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER200k50_CHNNwC_N800",
        "version_0",
        "epoch=14.ckpt"
        # "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

def save_paper_26():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "Gyro_homo_cor1_mu0_CHNNwC_N800",
        "version_1",
        "epoch=968.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=3)
    assert 1

def save_paper_27():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "Gyro_homo_cor0.8_mu0.1_CHNNwC_N800",
        "version_1",
        "epoch=892.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=3)
    assert 1

def save_paper_28():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "Cloth_8K200_CHNNwC_N800",
        "version_0",
        # "epoch=24.ckpt"
        "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)
    assert 1

def save_paper_29():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER200k50_CHNNwC_N800",
        "version_2",
        # "epoch=14.ckpt"
        "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

def save_paper_291():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER_default_CHNNwC_N800",
        "version_2",
        # "epoch=14.ckpt"
        "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

#########################################
# CHNNwC_mujoco
def save_paper_30():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BM5_homo_cor1_mu0_g0_mujoco_CHNNwC_N800",
        "version_0",
        "epoch=997.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)

def save_paper_31():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BM5_hetero_g0_mujoco_CHNNwC_N800",
        "version_0",
        "epoch=870.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)

def save_paper_32():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BD5_hetero_g0_mujoco_CHNNwC_N800",
        "version_0",
        "epoch=990.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=2)
    assert 1

def save_paper_33():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "CP3_ground_homo_cor1_mu0_mujoco_CHNNwC_N800",
        "version_0",
        "epoch=948.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

def save_paper_34():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "CP3_ground_homo_cor0_mu0.5_mujoco_CHNNwC_N800",
        "version_0",
        "epoch=981.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

def save_paper_35():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER200k50_mujoco_CHNNwC_N800",
        "version_0",
        "epoch=8.ckpt"
        # "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

def save_paper_36():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "Gyro_homo_cor1_mu0_mujoco_CHNNwC_N800",
        "version_0",
        "epoch=856.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=3)
    assert 1

def save_paper_37():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "Gyro_homo_cor0.8_mu0.1_mujoco_CHNNwC_N800",
        "version_0",
        "epoch=665.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=3)
    assert 1

def save_paper_38():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "Cloth_8K200_mujoco_CHNNwC_N800",
        "version_0",
        # "epoch=26.ckpt"
        "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)
    assert 1

def save_paper_39():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER200k50_mujoco_CHNNwC_N800",
        "version_2",
        # "epoch=8.ckpt"
        "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

def save_paper_391():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER_default_mujoco_CHNNwC_N800",
        "version_2",
        # "epoch=8.ckpt"
        "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

########
# CLNN_CD_MLP
def save_paper_40():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BM5_homo_cor1_mu0_g0_CLNN_CD_MLP_N800",
        "version_0",
        "epoch=812.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)

def save_paper_41():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BM5_hetero_g0_CLNN_CD_MLP_N800",
        "version_0",
        "epoch=959.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)

def save_paper_42():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BD5_hetero_g0_CLNN_CD_MLP_N800",
        "version_0",
        "epoch=919.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=2)
    assert 1

def save_paper_43():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "CP3_ground_homo_cor1_mu0_CLNN_CD_MLP_N800",
        "version_0",
        "epoch=997.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

def save_paper_44():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "CP3_ground_homo_cor0_mu0.5_CLNN_CD_MLP_N800",
        "version_0",
        "epoch=890.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

def save_paper_45():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER200k50_CLNN_CD_MLP_N800",
        "version_0",
        "epoch=894.ckpt"
        # "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

def save_paper_46():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "Gyro_homo_cor1_mu0_CLNN_CD_MLP_N800",
        "version_0",
        "epoch=933.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=3)
    assert 1

def save_paper_47():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "Gyro_homo_cor0.8_mu0.1_CLNN_CD_MLP_N800",
        "version_0",
        "epoch=858.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=3)
    assert 1

def save_paper_48():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "Cloth_8K200_CLNN_CD_MLP_N800",
        "version_0",
        # "epoch=984.ckpt"
        "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)
    assert 1

########################
# CLNN_IN
def save_paper_50():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BM5_homo_cor1_mu0_g0_CLNN_IN_N800",
        "version_0",
        "epoch=543.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)

def save_paper_51():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BM5_hetero_g0_CLNN_IN_N800",
        "version_0",
        "epoch=810.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)

def save_paper_52():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BD5_hetero_g0_CLNN_IN_N800",
        "version_0",
        "epoch=955.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=2)
    assert 1

def save_paper_53():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "CP3_ground_homo_cor1_mu0_CLNN_IN_N800",
        "version_0",
        "epoch=850.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

def save_paper_54():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "CP3_ground_homo_cor0_mu0.5_CLNN_IN_N800",
        "version_0",
        "epoch=986.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

def save_paper_55():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER200k50_CLNN_IN_N800",
        "version_0",
        "epoch=997.ckpt"
        # "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

def save_paper_56():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "Gyro_homo_cor1_mu0_CLNN_IN_N800",
        "version_0",
        "epoch=954.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=3)
    assert 1

def save_paper_57():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "Gyro_homo_cor0.8_mu0.1_CLNN_IN_N800",
        "version_0",
        "epoch=954.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=3)
    assert 1

def save_paper_58():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "Cloth_8K200_CLNN_IN_N800",
        "version_0",
        # "epoch=950.ckpt"
        "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)
    assert 1

########################################3
# IN
def save_paper_60():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BM5_homo_cor1_mu0_g0_IN_N800",
        "version_0",
        "epoch=1551.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)

def save_paper_61():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BM5_hetero_g0_IN_N800",
        "version_0",
        "epoch=1468.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)

def save_paper_62():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BD5_hetero_g0_IN_N800",
        "version_0",
        "epoch=1107.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=2)
    assert 1

def save_paper_63():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "CP3_ground_homo_cor1_mu0_IN_N800",
        "version_0",
        "epoch=1967.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

def save_paper_64():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "CP3_ground_homo_cor0_mu0.5_IN_N800",
        "version_0",
        "epoch=1952.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

def save_paper_65():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER200k50_IN_N800",
        "version_0",
        "epoch=903.ckpt"
        # "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=0)
    assert 1

def save_paper_66():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "Gyro_homo_cor1_mu0_IN_N800",
        "version_0",
        "epoch=1944.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=3)
    assert 1

def save_paper_67():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "Gyro_homo_cor0.8_mu0.1_IN_N800",
        "version_0",
        "epoch=1956.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=3)
    assert 1

def save_paper_68():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "Cloth_8K200_IN_N800",
        "version_0",
        # "epoch=891.ckpt"
        "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)
    assert 1

def save_paper_70():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "CP3_ground_homo_cor1_mu0_mujoco_CLNNwC_N800",
        "version_6",
        "epoch=978.ckpt"
        # "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)

def save_paper_71():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "CP3_ground_homo_cor0_mu0.5_mujoco_CLNNwC_N800",
        "version_6",
        # "epoch=926.ckpt"
        "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)

def save_paper_72():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "BM5_homo_cor1_mu0_g0_mujoco_CLNNwC_N800",
        "version_4",
        "epoch=991.ckpt"
        # "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)

def save_paper_73():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "Gyro_homo_cor0.8_mu0.1_mujoco_CLNNwC_N800",
        "version_10",
        "epoch=991.ckpt"
        # "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model, n_p_1=3)
# rope_extra 50k50
def save_paper_80():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER50k50_CLNNwC_N800",
        "version_0",
        "epoch=605.ckpt"
        # "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)

def save_paper_81():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER50k50_CLNN_CD_MLP_N800",
        "version_1",
        "epoch=980.ckpt"
        # "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)

def save_paper_82():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER50k50_CLNN_IN_N800",
        "version_0",
        "epoch=754.ckpt"
        # "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)

def save_paper_83():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER50k50_IN_N800",
        "version_0",
        "epoch=843.ckpt"
        # "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)
#rope_extra 100k50
def save_paper_84():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER100k50_CLNNwC_N800",
        "version_1",
        "epoch=200.ckpt"
        # "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)

def save_paper_85():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER100k50_CLNN_CD_MLP_N800",
        "version_0",
        "epoch=994.ckpt"
        # "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)

def save_paper_86():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER100k50_CLNN_IN_N800",
        "version_0",
        "epoch=942.ckpt"
        # "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)

def save_paper_87():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER100k50_IN_N800",
        "version_0",
        "epoch=886.ckpt"
        # "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)
# rope_extra 200k50 large need to change the name!
def save_paper_88():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER200k50_CLNNwC_N800",
        "version_3",
        "epoch=234.ckpt"
        # "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)

def save_paper_89():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER200k50_CLNN_CD_MLP_N800",
        "version_2",
        "epoch=987.ckpt"
        # "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)
    
def save_paper_90():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER200k50_CLNN_IN_N800",
        "version_1",
        "epoch=990.ckpt"
        # "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)

def save_paper_91():
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "ER200k50_IN_N800",
        "version_2",
        "epoch=705.ckpt"
        # "last.ckpt"
    ) 
    model = Model.load_from_checkpoint(checkpoint_path)
    utils(model)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n", type=str, required=True)
    args = parser.parse_args()
    func_name = "save_paper_" + args.n
    str_to_class(func_name)()

