from torch.utils.data import Dataset
import torch
import os

from systems.bouncing_mass_points import BouncingMassPoints
def rel_err(x, y):
    return (((x - y) ** 2).sum() / ((x + y) ** 2).sum()).sqrt()

class RigidBodyDataset(Dataset):
    def __init__(
        self, 
        root_dir=os.path.dirname(os.path.abspath(__file__)),
        body=BouncingMassPoints(1),
        n_traj=100,
        mode="train", 
        dtype=torch.float32,
        chunk_len=5,
        regen=False
    ):
        super().__init__()
        self.mode = mode
        self.body = body
        filename = os.path.join(
            root_dir, f"traj_{body}_N{n_traj}_{mode}.pt"
        )
        if os.path.exists(filename) and not regen:
            ts, zs = torch.load(filename)
        else:
            ts, zs = self.generate_trajectory_data(n_traj)
            os.makedirs(root_dir, exist_ok=True)
            torch.save((ts, zs), filename)
        ts, zs = self.chunk_training_data(ts, zs, chunk_len)

        self.ts, self.zs = ts.to(dtype=dtype), zs.to(dtype=dtype)
    
    def __len__(self):
        return self.zs.shape[0]

    def __getitem__(self, idx):
        return (self.zs[idx, 0], self.ts[idx]), self.zs[idx]

    def generate_trajectory_data(self, n_traj):
        """
        return ts, zs
        ts: n_traj, traj_len
        zs: n_traj, traj_len, z_dim
        """
        z0s = self.body.sample_initial_conditions(n_traj)
        ts = torch.arange(
            0, self.body.integration_time, self.body.dt, device=z0s.device, dtype=z0s.dtype
        )
        zs = self.body.integrate(z0s, ts)
        ts = ts.repeat(n_traj, 1)
        return ts, zs

    def chunk_training_data(self, ts, zs, chunk_len):
        """ Randomly samples chunks of trajectory data, returns tensors shaped for training.
        Inputs: [ts (batch_size, traj_len)] [zs (batch_size, traj_len, *z_dim)]
        outputs: [chosen_ts (batch_size, chunk_len)] [chosen_zs (batch_size, chunk_len, *z_dim)]"""
        n_trajs, traj_len, *z_dim = zs.shape
        n_chunks = traj_len // chunk_len
        # Cut each trajectory into non-overlapping chunks
        chunked_ts = torch.stack(ts.chunk(n_chunks, dim=1))
        chunked_zs = torch.stack(zs.chunk(n_chunks, dim=1))
        # From each trajectory, we choose a single chunk randomly
        chunk_idx = torch.randint(0, n_chunks, (n_trajs,), device=zs.device).long()
        chosen_ts = chunked_ts[chunk_idx, range(n_trajs)]
        chosen_zs = chunked_zs[chunk_idx, range(n_trajs)]
        return chosen_ts, chosen_zs
