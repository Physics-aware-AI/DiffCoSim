import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from datasets.datasets import RigidBodyDataset
from systems.billiards import Billiards
from pytorch_lightning import seed_everything

import torch
import torch.nn as nn
import json
from trainer import Model

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

if __name__ == "__main__":
    test_billiards()