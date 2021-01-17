import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dynamics import ConstrainedLagrangianDynamics, LagrangianDynamics, DeLaNDynamics
from utils import Linear, mlp, Reshape, CosSin
from systems.rigid_body import EuclideanT, GeneralizedT, rigid_DPhi
from .hamiltonian import HNN_Struct
from torchdiffeq import odeint
from typing import Optional, Union, Tuple
from models.impulse import ImpulseSolver

class CLNNwC(nn.Module):
    def __init__(self, body_graph, impulse_solver, n_c, d, 
        is_homo=False,
        hidden_size: int = 256,
        num_layers: int = 3,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        super().__init__()
        self.body_graph = body_graph
        self.n = len(body_graph.nodes)
        self.d, self.n_c = d, n_c
        self.dtype = dtype
        self.nfe = 0
        self.dynamics = ConstrainedLagrangianDynamics(self.potential, self.Minv_op, 
                                                      self.DPhi, (self.n, self.d))
        sizes = [self.n * self.d] + num_layers * [hidden_size] + [1]
        self.V_net = nn.Sequential(
            mlp(sizes, nn.Softplus, orthogonal_init=True), 
            Reshape(-1)
        )
        self.m_params = nn.ParameterDict(
            {str(d): nn.Parameter(0.1 * torch.randn(len(ids)//(d+1), d+1, dtype=dtype)) # N, d+1
                for d, ids in body_graph.d2ids.items()}
        )
        assert len(self.m_params) == 1 # limited support for now
        self.n_p = int(list(self.m_params.keys())[0]) + 1
        self.n_o = self.n // self.n_p
        if is_homo:
            self.mu_params = nn.Parameter(torch.rand(1, dtype=dtype))
            self.cor_params = nn.Parameter(torch.randn(1, dtype=dtype))
        else:
            self.mu_params = nn.Parameter(torch.rand(n_c, dtype=dtype))
            self.cor_params = nn.Parameter(torch.randn(n_c, dtype=dtype))
        self.is_homo = is_homo

        self.impulse_solver = impulse_solver

    @property
    def Minv(self):
        n = self.n
        # d == n_p-1
        d = int(list(self.m_params.keys())[0])
        inv_moments = torch.exp(-self.m_params[str(d)]) # n_o, n_p
        inv_masses = inv_moments[:, :1] # n_o, 1
        if d == 0:
            return torch.diag_embed(inv_masses[:, 0])
        blocks = torch.diag_embed(torch.cat([0*inv_masses, inv_moments[:, 1:]], dim=-1)) # n_o, n_p, n_p
        blocks = blocks + torch.ones_like(blocks)
        blocks = blocks * inv_masses.unsqueeze(-1)
        return torch.block_diag(*blocks) # (n, n)

    @property
    def L_Minv(self):
        n = self.n
        # d == n_p-1
        d = int(list(self.m_params.keys())[0])
        inv_moments_sqrt = torch.exp(-self.m_params[str(d)]/2) # n_o, n_p
        inv_masses_sqrt = inv_moments_sqrt[:, :1] # n_o, 1
        if d == 0:
            return torch.diag_embed(inv_masses_sqrt[:, 0])
        blocks = torch.diag_embed(torch.cat([0*inv_masses_sqrt, inv_moments_sqrt[:, 1:]], dim=-1)) # n_o, n_p, n_p
        blocks[:, :, 0] = torch.ones_like(blocks[:, :, 0])
        blocks = blocks * inv_masses_sqrt.unsqueeze(-1)
        return torch.block_diag(*blocks) # (n, n)

    def L_Minv_op(self):
        pass

    def Minv_op(self, p):
        assert len(self.m_params) == 1 
        *dims, n, a = p.shape
        d = int(list(self.m_params.keys())[0])
        n_o = n // (d+1) # number of extended bodies
        p_reshaped = p.reshape(*dims, n_o, d+1, a)
        inv_moments = torch.exp(-self.m_params[str(d)])
        inv_masses = inv_moments[:, :1] # n_o, 1
        if d == 0:
            return (inv_masses.unsqueeze(-1)*p_reshaped).reshape(*p.shape)
        inv_mass_p = inv_masses.unsqueeze(-1) * p_reshaped.sum(-2, keepdims=True) # (n_o, 1, 1) * (..., n_o, 1, a) = (..., n_o, 1, a)
        padded_intertias_inv = torch.cat([0*inv_masses, inv_moments[:, 1:]], dim=-1) # (n_o, d+1)
        inv_intertia_p = padded_intertias_inv.unsqueeze(-1) * p_reshaped # (n_o, d+1, 1) * (..., n_o, d+1, a) = (..., n_o, d+1, a)
        return (inv_mass_p + inv_intertia_p).reshape(*p.shape)

    def forward(self, t, z):
        self.nfe += 1
        return self.dynamics(t, z)

    def DPhi(self, x, x_dot):
        return rigid_DPhi(self.body_graph, x, x_dot)

    def potential(self, x):
        assert x.ndim == 3
        return self.V_net(x.reshape(x.shape[0], -1))

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
        bs = z0.shape[0]
        if self.is_homo:
            mus = F.relu(self.mu_params * torch.ones(self.n_c).type_as(self.mu_params))
            cors = F.hardsigmoid(self.cor_params* torch.ones(self.n_c).type_as(self.cor_params))
        else:
            mus = F.relu(self.mu_params)
            cors = F.hardsigmoid(self.cor_params)
        ts = ts.to(z0.device, z0.dtype)
        zt = z0.reshape(bs, -1)
        zT = torch.zeros([bs, len(ts), zt.shape[1]], device=z0.device, dtype=z0.dtype)
        zT[:, 0] = zt
        for i in range(len(ts)-1):
            zt_n = odeint(self, zt, ts[i:i+2], rtol=tol, method=method)[1]
            zt_n, _ = self.impulse_solver.add_impulse(zt_n, mus, cors, self.Minv)
            zt = zt_n
            zT[:, i+1] = zt
        return zT.reshape(bs, len(ts), 2, self.n, self.d)


class LNN_Struct(HNN_Struct):
    """ DeLaN """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamics = DeLaNDynamics(self.V_net, self.M, self.Minv)

    def integrate(self, z0, ts, tol=1e-4, method="rk4"):
        """
        Input:
            z0: (N, 2, D)
            ts: (T,)
        Returns: (N, T, 2, D)
        """
        assert (z0.ndim == 3) and (ts.ndim == 1)
        assert z0.shape[-1] == self.q_ndim
        bs, _, D = z0.shape
        zT = odeint(self, z0.reshape(bs, -1), ts, rtol=tol, method=method).permute(1, 0, 2)
        return zT.reshape(bs, len(ts), 2, D)


class LNN_Struct_Angle(LNN_Struct):
    def __init__(self, body_graph, dof_ndim, num_layers, 
                    hidden_size, dtype, angular_dims, **kwargs):
        super().__init__(body_graph, dof_ndim, num_layers, hidden_size, dtype, **kwargs)

        self.angular_dims = angular_dims
        sizes = [self.q_ndim + len(self.angular_dims)] + num_layers * [hidden_size] + [1]
        self.V_net = nn.Sequential(
            CosSin(self.q_ndim, self.angular_dims, only_q=True),
            mlp(sizes, nn.Softplus, orthogonal_init=True),
            Reshape(-1)
        )
        sizes = [self.q_ndim + len(self.angular_dims)] + num_layers * [hidden_size] + [self.q_ndim*self.q_ndim]
        self.M_net = nn.Sequential(
            CosSin(self.q_ndim, self.angular_dims, only_q=True),
            mlp(sizes, nn.Softplus, orthogonal_init=True),
            Reshape(-1, self.q_ndim, self.q_ndim)
        )


class LNN(nn.Module):
    def __init__(self, body_graph, dof_ndim, num_layers, hidden_size, dtype, 
                    **kwargs):
        super().__init__()
        # self.body_graph = body_graph
        self.nfe = 0
        self.q_ndim = dof_ndim
        self.dynamics = LagrangianDynamics(self.lagrangian)
        sizes = [2*self.q_ndim] + num_layers * [2*hidden_size] + [1]
        self.L_net = nn.Sequential(
            mlp(sizes, nn.Softplus, orthogonal_init=True), 
            Reshape(-1)
        )

    def lagrangian(self, t, z, eps=0.1):
        # Add regularization to prevent singular mass matrix at initialization
        # equivalent to adding eps to the diagonal of the mass matrix (Hessian of L)
        # Note that the network could learn to offset this added term
        q, qdot = z.chunk(2, dim=-1)
        reg = eps * (qdot * qdot).sum(-1)
        return self.L_net(z) + reg

    def forward(self, t, z):
        self.nfe += 1
        return self.dynamics(t, z)

    def integrate(self, z0, ts, tol=1e-4, method="rk4"):
        """
        Input:
            z0: (N, 2, D)
            ts: (T,)
        Returns: (N, T, 2, D)
        """
        assert (z0.ndim == 3) and (ts.ndim == 1)
        assert z0.shape[-1] == self.q_ndim
        bs, _, D = z0.shape
        zT = odeint(self, z0.reshape(bs, -1), ts, rtol=tol, method=method).permute(1, 0, 2)
        return zT.reshape(bs, len(ts), 2, D)


class LNN_Angle(LNN):
    def __init__(self, body_graph, dof_ndim, num_layers, hidden_size, dtype, 
                    angular_dims, **kwargs):
        super().__init__(body_graph, dof_ndim, num_layers, hidden_size, dtype, 
                    **kwargs)
        sizes = [2*self.q_ndim + len(angular_dims)] + num_layers * [2*hidden_size] + [1]
        self.L_net = nn.Sequential(
            CosSin(self.q_ndim, angular_dims, only_q=False),
            mlp(sizes, nn.Softplus, orthogonal_init=True), 
            Reshape(-1)
        )
        