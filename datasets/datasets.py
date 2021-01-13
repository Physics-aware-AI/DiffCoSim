from torch.utils.data import Dataset
import torch
import os
from pytorch_lightning import seed_everything

from systems.bouncing_mass_points import BouncingMassPoints
def rel_err(x, y):
    return (((x - y) ** 2).sum() / ((x + y) ** 2).sum()).sqrt()

class RigidBodyDataset(Dataset):
    def __init__(
        self, 
        root_dir=os.path.dirname(os.path.abspath(__file__)),
        body=BouncingMassPoints(),
        n_traj=100,
        mode="train", 
        dtype=torch.float32,
        chunk_len=5,
        regen=False,
    ):
        super().__init__()
        self.mode = mode
        self.body = body
        filename = os.path.join(
            root_dir, f"traj_{body}_N{n_traj}_{mode}.pt"
        )
        if os.path.exists(filename) and not regen:
            ts, zs, is_clds = torch.load(filename)
        else:
            print(f"generating trajectories (mode: {mode}), this might take a while...")
            seed_everything(0)
            ts, zs, is_clds = self.generate_trajectory_data(n_traj)
            os.makedirs(root_dir, exist_ok=True)
            torch.save((ts, zs, is_clds), filename)
        seed_everything(0)
        ts, zs, is_clds = self.chunk_training_data(ts, zs, is_clds, chunk_len)

        self.ts, self.zs = ts.to(dtype=dtype), zs.to(dtype=dtype)
        print(f"{is_clds.sum()} out of {len(is_clds)} trajectories contains collision.")
    
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
        zs, is_clds = self.body.integrate(z0s, ts)
        ts = ts.repeat(n_traj, 1)
        return ts, zs, is_clds

    def chunk_training_data(self, ts, zs, is_clds, chunk_len):
        """ Randomly samples chunks of trajectory data, returns tensors shaped for training.
        Inputs: [ts (bs, traj_len)] [zs (bs, traj_len, *z_dim)] [is_clds (bs, traj_len)]
        outputs: [chosen_ts (bs, chunk_len)] [chosen_zs (bs, chunk_len, *z_dim)]"""
        n_trajs, traj_len, *z_dim = zs.shape
        n_chunks = traj_len // chunk_len
        # Cut each trajectory into non-overlapping chunks
        chunked_ts = torch.stack(ts.chunk(n_chunks, dim=1))
        chunked_zs = torch.stack(zs.chunk(n_chunks, dim=1))
        chunked_is_clds = torch.stack(is_clds.chunk(n_chunks, dim=1)) # n_chunks, bs, chunk_len
        is_clds_t0 = chunked_is_clds[..., 0] # n_chunks, bs
        is_cld = chunked_is_clds.sum(dim=-1) # n_chunks, bs
        # From each trajectory, we choose a single chunk randomly
        # we make sure that the initial condition is not during collision
        chosen_ts = torch.zeros(n_trajs, chunk_len, dtype=ts.dtype, device=ts.device)
        chosen_zs = torch.zeros(n_trajs, chunk_len, *chunked_zs.shape[3:], dtype=zs.dtype, device=zs.device)
        is_cld_in_chosen = torch.zeros(n_trajs, dtype=torch.bool, device=zs.device)
        for i in range(n_trajs):
            no_cld0_idx = torch.nonzero(is_clds_t0[:, i] == 0, as_tuple=False)[:, 0]
            rand_idx = torch.randint(0, len(no_cld0_idx), (1,), device=zs.device)[0]
            chosen_ts[i, :] = chunked_ts[no_cld0_idx[rand_idx], i]
            chosen_zs[i, :] = chunked_zs[no_cld0_idx[rand_idx], i]
            is_cld_in_chosen[i] = chunked_is_clds[no_cld0_idx[rand_idx], i].sum() > 0
        return chosen_ts, chosen_zs, is_cld_in_chosen
