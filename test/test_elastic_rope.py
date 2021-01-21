import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from datasets.datasets import RigidBodyDataset
from systems.elastic_rope import ElasticRope
from pytorch_lightning import seed_everything

import torch
import torch.nn as nn
import json
from trainer import Model

seed_everything(0)

import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")

def test_elastic_rope_0():
    body = ElasticRope()
    dataset = RigidBodyDataset(
        mode = "test",
        n_traj = 10,
        body = body,
        dtype = torch.float32,
        chunk_len = 100,
        regen=True
    )

    ani = body.animate(dataset.zs, 3)
    ani.save(os.path.join(THIS_DIR, 'test_elastic_rope_0.gif'))

    assert 1

if __name__ == "__main__":
    test_elastic_rope_0()