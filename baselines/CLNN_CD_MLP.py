from models.lagrangian import CLNNwC
import torch
import torch.nn as nn
from utils import mlp
from torchdiffeq import odeint

class CLNN_CD_MLP(CLNNwC):
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
        self.check_collision = self.impulse_solver.check_collision 
        self.impulse_solver = None
        sizes = [2*self.n*self.d] + num_layers * [hidden_size] + [self.n*self.d]
        self.velocity_impulse = mlp(sizes, nn.ReLU, orthogonal_init=True)


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
        # delta_v = self.velocity_impulse(z1)
        z1 = z1.reshape(bs, 2, self.n, self.d)
        x1 = z1[:, 0] ; v1 = z1[:, 1]
        is_cld, is_cld_ij, is_cld_bdry, is_cld_limit, dist_ij, dist_bdry, dist_limit= self.check_collision(x1)
        delta_v = torch.zeros(bs, self.n, self.d).type_as(z0)
        if is_cld.sum() != 0:
            delta_v_sub = self.velocity_impulse(z1[is_cld].reshape(is_cld.sum(), -1))
            delta_v[is_cld] = delta_v_sub.reshape(is_cld.sum(), self.n, self.d)
        v1 = v1 + delta_v.reshape(bs, self.n, self.d)
        z1 = torch.stack([x1, v1], dim=1)
        return torch.stack([z0, z1], dim=1)
