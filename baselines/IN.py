from models.lagrangian import CLNNwC
import torch
import torch.nn as nn
from utils import mlp
from torchdiffeq import odeint
from .interaction_network import InteractionNetwork

class IN(CLNNwC):
    def __init__(
        self,
        body_graph,
        impulse_solver,
        n_c,
        d,
        is_homo=False,
        hidden_size: int = 256,
        num_layers: int = 3,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32,
        body=None,
        R_net_hidden_size=300,
        O_net_hidden_size=200,
        **kwargs
    ):
        super().__init__(
            body_graph,
            impulse_solver,
            n_c,
            d,
            is_homo,
            hidden_size,
            num_layers,
            device,
            dtype,
            **kwargs 
        )
        self.mu_params = None
        self.cor_params = None
        self.impulse_solver = None
        self.body = body
        self.velocity_impulse = InteractionNetwork(body, R_net_hidden_size, O_net_hidden_size)
        is_mass_point = "BM" in self.body.kwargs_file_name or \
                        "CP" in self.body.kwargs_file_name or \
                        "ER" in self.body.kwargs_file_name
        if is_mass_point:
            self.inv_moments = torch.div(1, self.body.ms)
        elif "BD" in self.body.kwargs_file_name:
            ms = self.body.ms
            ls = self.body.ls
            moments = ms * ls *ls / 4
            m_and_m = torch.stack([ms, moments, moments], dim=1)
            self.inv_moments = torch.div(1, m_and_m.reshape(-1))
        elif "Gyro" in self.body.kwargs_file_name:
            m = self.body.m    
            m_and_m = torch.cat(
                [torch.tensor(m).reshape(1).type_as(self.body.moments), m*self.body.moments],
                dim=0
            )
            self.inv_moments = torch.div(1, m_and_m)

    def get_f_external(self, bs, n, d):
        # f_external (bs, n, d)
        if "BM" in self.body.kwargs_file_name:
            return torch.zeros(bs, n, d)
        elif "BD" in self.body.kwargs_file_name:
            return torch.zeros(bs, n, d)
        elif "CP" in self.body.kwargs_file_name:
            inv_moments = torch.div(1, self.body.ms)
            g = self.body.g
            f_external = torch.stack(
                [torch.zeros(bs, n), -g * self.body.ms * torch.ones(bs, n)],
                dim=-1
            )
            return f_external
        elif "ER" in self.body.kwargs_file_name:
            g = self.body.g
            inv_moments = torch.div(1, self.body.ms)
            f_external = torch.stack(
                [torch.zeros(bs, n), -g * self.body.ms * torch.ones(bs, n)],
                dim=-1
            )
            return f_external
        elif "Gyro" in self.body.kwargs_file_name:
            f_external = torch.zeros(bs, 4, 3)
            f_external[:, 0, 2] = f_external[:, 0, 2] - self.body.m * 9.81
            return f_external

    def integrate(self, z0, ts, tol=1e-4, method="rk4"):
        """
        input:
            z0: bs, 2, n, d
            ts: length T
        returns:
            a tensor of size bs, T, 2, n, d
        """
        assert (z0.ndim == 4) and (ts.ndim == 1)
        assert (z0.shape[-1] == self.d) and z0.shape[-2] == self.n
        assert len(ts) == 2

        bs, _, n, d = z0.shape
        # true Minv
        f_external = self.get_f_external(bs, n, d)

        v1 = self.velocity_impulse(z0, self.inv_moments.type_as(z0), f_external) # bs, n, d
        x0 = z0[:, 0] ; v0 = z0[:, 1]
        x1 = x0 + v0 * (ts[1] - ts[0])
        z1 = torch.stack([x1, v1], dim=1)
        return torch.stack([z0, z1], dim=1)