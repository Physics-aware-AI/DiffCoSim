import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)
import json

from datasets.datasets import RigidBodyDataset
from systems.bouncing_mass_points import BouncingMassPoints
from systems.chain_pendulum_with_contact import ChainPendulum_w_Contact
from pytorch_lightning import seed_everything
from models.impulse import ImpulseSolver

import torch

seed_everything(0)

import matplotlib.pyplot as plt
plt.switch_backend("Agg")

def test_BM():
    body = BouncingMassPoints()
    # initial condition 
    z0 = torch.zeros([1, 2, 1, 2], dtype=torch.float64)
    z0[0, 0, 0, 0] = 0.5 # x1
    z0[0, 0, 0, 1] = -0.2 # x2
    z0[0, 1, 0, 0] = 0.1 # v1
    z0[0, 1, 0, 1] = - z0[0, 0, 0, 1] / body.dt # v2

    ts = torch.arange(0, 5*body.dt, body.dt, device=z0.device, dtype=z0.dtype)
    zs = body.integrate(z0, ts)
    ani = body.animate(zs, 0)
    ani.save(os.path.join(THIS_DIR, "test_c.gif"))
    # make sure ball is bouncing back
    assert zs[0, 1, 0, 0, 1] > zs[0, 0, 0, 0, 1] 

def test_corner():
    body = BouncingMassPoints(
        ms=[1/0.9564],
        mus=[1.6585, 0.7067, 0.8111, 0.6014],
        cors=[0.6449, 0.6264, 0.2586, 0.4543],
    )
    z0 = torch.zeros(1, 2, 1, 2, dtype=torch.float32)
    z0[0, 0, 0, 0] = 0.1018
    z0[0, 0, 0, 1] = 0.1226
    z0[0, 1, 0, 0] = -1.5953
    z0[0, 1, 0, 1] = -2.4683

    ts = torch.arange(0, 10*body.dt, body.dt).type_as(z0)
    zs, _ = body.integrate(z0, ts)
    ani = body.animate(zs, 0)
    ani.save(os.path.join(THIS_DIR, "test_corner.gif"))

def test_load_saved_tensor():
    body = BouncingMassPoints(
        ms=[1/0.9564],
        mus=[1.6585, 0.7067, 0.8111, 0.6014],
        cors=[0.6449, 0.6264, 0.2586, 0.4543],
    )
    tensors = torch.load(os.path.join(PARENT_DIR, "test", "tensors_0", "rest.pt"))
    if len(tensors) == 8:
        bs_idx, v, Minv, Jac, Jac_v, v_star, mu, cor = tensors
        body.impulse_solver.get_dv_wo_J_e(*tensors)

def test_load_saved_tensor_0():
    body_kwargs_file = "BM3_homo_cor1_mu0"
    with open(os.path.join(PARENT_DIR, "examples", body_kwargs_file + ".json")) as file:
        body_kwargs = json.load(file)
    body = BouncingMassPoints(body_kwargs_file, **body_kwargs)
    tensors = torch.load(os.path.join(PARENT_DIR, "tensors", "tensors_0", "rest.pt"))
    if len(tensors) == 8:
        bs_idx, v, Minv, Jac, Jac_v, v_star, mu, cor = tensors
        body.impulse_solver.get_dv_wo_J_e(*tensors)
    else:
        bs_idx, v, Minv, Jac, Jac_v, v_star, mu, cor, DPhi = tensors
        body.impulse_solver.get_dv_w_J_e(*tensors)


def test_load_saved_tensor_1():
    body_kwargs_file = "CP1_wall_homo_cor1_mu1"
    with open(os.path.join(PARENT_DIR, "examples", body_kwargs_file + ".json")) as file:
        body_kwargs = json.load(file)
    body = ChainPendulum_w_Contact(body_kwargs_file, **body_kwargs)
    tensors = torch.load(os.path.join(PARENT_DIR, "tensors", "tensors_610", "comp.pt"))
    if len(tensors) == 8:
        bs_idx, v, Minv, Jac, Jac_v, v_star, mu, cor = tensors
        body.impulse_solver.get_dv_wo_J_e(*tensors)
    else:
        bs_idx, v, Minv, Jac, Jac_v, v_star, mu, cor, DPhi = tensors
        body.impulse_solver.get_dv_w_J_e(*tensors)

if __name__ == "__main__":
    test_load_saved_tensor_1()