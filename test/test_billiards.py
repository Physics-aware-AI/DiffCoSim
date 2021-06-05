import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from datasets.datasets import RigidBodyDataset
from systems.billiards import Billiards, BilliardsDummyAnimation
from pytorch_lightning import seed_everything

import torch
import torch.nn as nn
import json
# from trainer import Model
from trainer_billiards import Model

import matplotlib.animation as animation


seed_everything(0)

import matplotlib.pyplot as plt
plt.switch_backend("Agg")

def test_billiards():
    body = Billiards()
    dataset = RigidBodyDataset(
        mode = "test",
        n_traj = 1,
        body = body,
        dtype = torch.float32,
        chunk_len = 1024,
        regen=False
    )

    ani = body.animate(dataset.zs, 0)
    ani.save(os.path.join(THIS_DIR, 'test_billiards.gif'), writer="pillow")

def test_billiards_learned():
    body = Billiards()
    checkpoint_path = os.path.join(
        PARENT_DIR,
        "logs",
        "billiards",
        "version_3",
        "epoch=189.ckpt"
    ) 
    i = 100
    model = Model.load_from_checkpoint(checkpoint_path)
    init_cond = model.history[i]
    # model.initial_xy = nn.Parameter(torch.from_numpy(init_cond[0]))
    # model.initial_vxvy = nn.Parameter(torch.from_numpy(init_cond[1]))
    # model.initial_xy = nn.Parameter(torch.tensor([0.39771125, 0.47811887]))
    # model.initial_vxvy = nn.Parameter(torch.tensor([0.49173883, -0.00338323]))
    model.initial_xy = nn.Parameter(torch.tensor([0.44567704, 0.47560564]))
    model.initial_vxvy = nn.Parameter(torch.tensor([0.5000665, -0.00096817]))
    model.eval()
    zT, loss = model.test_step(None, None)
    ani = body.animate(zT.reshape(1, -1, 2, 11, 2), 0)
    writervideo = animation.FFMpegWriter(fps=30)
    ani.save(os.path.join(THIS_DIR, f'test_billiards_learned_difftaichi_{i}.mp4'), writer=writervideo)

def save_png():
    body = Billiards()

    # get default initial condition
    zs = body.get_initial_conditions() # (1, 2, n, 2)
    # modify the initial condition of the first ball into desired ones
    zs[0, :, 0, :] = torch.tensor([[0.4458184, 0.47530562],
                                   [0.5002306, -0.00089275]])
    # zs[0, :, 0, :] = torch.tensor([[-0.0023, 0.4218],
    #                                [0.5860, 0.0596]])

    animator = BilliardsDummyAnimation(zs[0:1, 0, :, :], body, zs[0, 1, 0, 0].item(), zs[0, 1, 0, 1].item())
    animator.update()
    animator.fig.savefig(os.path.join(THIS_DIR, 'billiards_learned_difftaichi.png'), bbox_inches='tight')
    # animator.fig.savefig(os.path.join(THIS_DIR, 'billiards_learned_our_model.png'), bbox_inches='tight')

if __name__ == "__main__":
    # test_billiards()
    # save_png()
    test_billiards_learned()