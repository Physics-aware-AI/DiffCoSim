from models.lagrangian import CLNNwC
import torch
import torch.nn as nn
from utils import mlp
from torchdiffeq import odeint
from .interaction_network import InteractionNetwork

class CLNN_IN(CLNNwC):
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
        R_net_hidden_size=150,
        O_net_hidden_size=100,
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

        bs = z0.shape[0]

        z1 = odeint(self, z0.reshape(bs, -1), ts, rtol=tol, method="rk4")[1]
        z1 = z1.reshape(bs, 2, self.n, self.d)
        x1 = z1[:, 0] ; v1 = z1[:, 1]
        # get mass
        # inv_moments = self.get_inv_moments()
        d = int(list(self.m_params.keys())[0])
        inv_moments = torch.exp(-self.m_params[str(d)]).reshape(-1) # n_o*n_p
        # add walls
        # get potential velocity
        V = self.potential(x1)
        dV = torch.autograd.grad(V.sum(), x1, create_graph=True)[0] # bs, n, d

        delta_v = self.velocity_impulse(z1, inv_moments, dV)
        v1 = v1 + delta_v.reshape(bs, self.n, self.d)
        z1 = torch.stack([x1, v1], dim=1)
        return torch.stack([z0, z1], dim=1)

    # def get_inv_moments(self):
    #     if "BM" in self.body.kwargs_file_name:
    #         return inv_moments
    #     elif "BD" in self.body.kwargs_file_name:
    #         d = int(list(self.m_params.keys())[0])
    #         inv_m = torch.exp(-self.m_params[str(d)]).reshape(-1) # n_o*n_p

    #         return inv_moments
