import os, sys

from networkx.algorithms.centrality.current_flow_betweenness_subset import edge_current_flow_betweenness_centrality_subset
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from datasets.datasets import RigidBodyDataset
from systems.cloth import Cloth
from pytorch_lightning import seed_everything
import networkx as nx

import torch
import torch.nn as nn
import json
from trainer import Model
import time

seed_everything(0)

import matplotlib.pyplot as plt
# plt.switch_backend("TkAgg")

def test_cloth_0():
    # body_kwargs_file = "default"
    # with open(os.path.join(PARENT_DIR, "examples", body_kwargs_file + ".json")) as file:
    #     body_kwargs = json.load(file)
    body = Cloth()
    # pos = nx.spring_layout(body.body_graph)
    # fig = plt.figure(figsize=(12,12))
    # ax = plt.subplot(111)
    # nx.draw(body.body_graph, pos, node_size=1500, node_color='yellow', font_size=8, font_weight='bold')
    # plt.savefig(os.path.join(THIS_DIR, 'test_cloth_0.png'))

    start_time = time.time()
    dataset = RigidBodyDataset(
        mode = "test",
        n_traj =1,
        body = body,
        dtype = torch.float32,
        chunk_len =20,
        regen=True,
        separate=True,
    )

    print(f"{time.time() - start_time} seconds")
    # N, T = dataset.zs.shape[:2]
    # x, v = dataset.zs.chunk(2, dim=2)
    # p_x = body.M.type_as(v) @ v
    # zts = torch.cat([x, p_x], dim=2)
    # energy = body.hamiltonian(None, zts.reshape(N*T, -1)).reshape(N, T)
    # plt.plot(energy[3])
    # plt.show()

    ani = body.animate(dataset.zs, 0)
    ani.save(os.path.join(THIS_DIR, 'test_cloth_0.gif'), writer="pillow")

    assert 1

if __name__ == "__main__":
    test_cloth_0()