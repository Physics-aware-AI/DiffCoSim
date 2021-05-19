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
        noise_std=0.0,
        separate=False,
    ):
        super().__init__()
        self.mode = mode
        self.body = body
        self.noise_std = noise_std
        self.separate = separate
        filename = os.path.join(
            root_dir, f"traj_{body}_N{n_traj}_{mode}.pt"
        )
        if os.path.exists(filename) and not regen:
            ts, zs, is_clds = torch.load(filename)
        else:
            print(f"generating trajectories (mode: {mode}), this might take a while...")
            seed_everything(0)
            ts, zs, is_clds = self.generate_trajectory_data(n_traj, separate=separate)
            os.makedirs(root_dir, exist_ok=True)
            torch.save((ts, zs, is_clds), filename)
        seed_everything(0)
        ts, zs, is_clds = self.chunk_training_data(ts, zs, is_clds, chunk_len)

        self.ts, self.zs, self.is_clds = ts.to(dtype=dtype), zs.to(dtype=dtype), is_clds
        print(f"{is_clds.sum()} out of {len(is_clds)} trajectories contains collision.")
    
    def __len__(self):
        return self.zs.shape[0]

    def __getitem__(self, idx):
        return (self.zs[idx, 0], self.ts[idx]), self.zs[idx], self.is_clds[idx]

    def generate_trajectory_data(self, n_traj, separate=False):
        """
        return ts, zs
        ts: n_traj, traj_len
        zs: n_traj, traj_len, z_dim
        """
        z0s = self.body.sample_initial_conditions(n_traj)
        ts = torch.arange(
            0, self.body.integration_time, self.body.dt, device=z0s.device, dtype=z0s.dtype
        )
        if not separate:
            zs, is_clds = self.body.integrate(z0s, ts)
            ts = ts.repeat(n_traj, 1)
            return ts, zs, is_clds
        else:
            zs, is_clds = [], []
            for i in range(n_traj):
                z, is_cld = self.body.integrate(z0s[i:i+1], ts)
                zs.append(z)
                is_clds.append(is_cld)
            ts = ts.repeat(n_traj, 1)
            return ts, torch.cat(zs, dim=0), torch.cat(is_clds, dim=0)

    def chunk_training_data(self, ts, zs, is_clds, chunk_len, p_cld=0.5):
        """ Randomly samples chunks of trajectory data, returns tensors shaped for training.
        Inputs: [ts (bs, T)] [zs (bs, T, *z_dim)] [is_clds (bs, T)]
        outputs: [chosen_ts (bs, chunk_len)] [chosen_zs (bs, chunk_len, *z_dim)]"""
        bs, T, *z_dim = zs.shape
        n_chunks = (T - chunk_len + 1)
        # Cut each trajectory into non-overlapping chunks
        chunked_ts = torch.stack([ts[:, i:i+chunk_len] for i in range(n_chunks)], dim=0) # n_chunks, bs, chunk_len
        chunked_zs = torch.stack([zs[:, i:i+chunk_len] for i in range(n_chunks)], dim=0) # n_chunks, bs, chunk_len, *z_dim
        chunked_is_clds = torch.stack([is_clds[:, i:i+chunk_len] for i in range(n_chunks)], dim=0) # n_chunks, bs, chunk_len
        is_clds_t0 = chunked_is_clds[..., 0] # n_chunks, bs
        is_clds_chunk = chunked_is_clds.sum(-1) > 0 # n_chunks, bs
        is_cld = chunked_is_clds.sum(dim=-1) # n_chunks, bs
        # From each trajectory, we choose a single chunk randomly
        # we make sure that the initial condition is not during collision
        chosen_ts = torch.zeros(bs, chunk_len, dtype=ts.dtype, device=ts.device)
        chosen_zs = torch.zeros(bs, chunk_len, *chunked_zs.shape[3:], dtype=zs.dtype, device=zs.device)
        is_cld_in_chosen = torch.zeros(bs, dtype=torch.bool, device=zs.device)
        # we make sure there are roughly p_cld trajectories that contains collision
        is_cld_T = is_clds.sum(-1) > 0
        cld_ratio = (is_cld_T).sum().true_divide(bs)
        if cld_ratio < p_cld:
            contains_cld = is_cld_T
        else:
            cld_idx = torch.nonzero(is_cld_T, as_tuple=False)[:, 0]
            rand_idx = torch.rand(len(cld_idx)) < p_cld / cld_ratio
            
            contains_cld = torch.zeros(bs, dtype=torch.bool, device=zs.device)
            contains_cld[cld_idx[rand_idx]] = True
        for i in range(bs):            
            no_cld0_no_cld_chunk_idx = torch.nonzero(
                torch.logical_and(is_clds_t0[:, i] == 0, is_clds_chunk[:, i] == 0), 
                as_tuple=False
            )[:, 0]
            no_cld0_cld_chunk_idx = torch.nonzero(
                torch.logical_and(is_clds_t0[:, i] == 0, is_clds_chunk[:, i] == 1), 
                as_tuple=False
            )[:, 0]
            if (contains_cld[i] and len(no_cld0_cld_chunk_idx) > 0) or len(no_cld0_no_cld_chunk_idx) == 0:
                rand_idx = torch.randint(0, len(no_cld0_cld_chunk_idx), (1,), device=zs.device)[0]
                chunk_idx = no_cld0_cld_chunk_idx[rand_idx]
            else:
                rand_idx = torch.randint(0, len(no_cld0_no_cld_chunk_idx), (1,), device=zs.device)[0]
                chunk_idx = no_cld0_no_cld_chunk_idx[rand_idx]
            chosen_ts[i, :] = chunked_ts[chunk_idx, i]
            chosen_zs[i, :] = chunked_zs[chunk_idx, i]
            is_cld_in_chosen[i] = chunked_is_clds[chunk_idx, i].sum() > 0
        if self.body.__class__.__name__ == "ChainPendulumWithContact":
            chosen_q_qdot = self.body.global_cartesian_to_angle(chosen_zs)
            chosen_q_qdot = chosen_q_qdot + torch.randn(*chosen_q_qdot.shape) * self.noise_std
            noisy_zs = self.body.angle_to_global_cartesian(chosen_q_qdot)
        else:
            noisy_zs = chosen_zs + torch.randn(*chosen_zs.shape) * self.noise_std
        return chosen_ts, noisy_zs, is_cld_in_chosen
