import torch
import torch.nn as nn
from .dynamics import ConstrainedLagrangianDynamics, LagrangianDynamics, DeLaNDynamics
from ..utils import Linear, mlp, Reshape, CosSin
from ..systems.rigid_body import EuclideanT, GeneralizedT, rigid_DPhi
from .hamiltonian import HNN_Struct
from torchdiffeq import odeint
from typing import Optional, Union, Tuple

class CLNN(nn.Module):
    def __init__(self, body_graph,
        dof_ndim: Optional[int] = None,
        angular_dims: Union[Tuple, bool] = tuple(),
        hidden_size: int = 256,
        num_layers = 3,
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        super().__init__()
        self.body_graph = body_graph
        self.n = len(body_graph.nodes)
        self.d = dof_ndim
        self.nfe = 0
        self.dynamics = ConstrainedLagrangianDynamics(self.potential, self.Minv, 
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

    def Minv(self, p):
        assert len(self.m_params) == 1 # limited support for now
        *dims, n, a = p.shape
        d = int(list(self.m_params.keys())[0])
        N = n // (d+1) # number of extended bodies
        p_reshaped = p.reshape(*dims, N, d+1, a)
        inv_moments = torch.exp(-self.m_params[str(d)])
        inv_masses = inv_moments[:, :1] # N, 1
        if d == 0:
            return (inv_masses.unsqueeze(-1)*p_reshaped).reshape(*p.shape)
        inv_mass_p = inv_masses.unsqueeze(-1) * p_reshaped.sum(-2, keepdims=True) # (N, 1, 1) * (..., N, 1, a) = (..., N, 1, a)
        padded_intertias_inv = torch.cat([0*inv_masses, inv_moments[:, 1:]], dim=-1) # (N, d+1)
        inv_intertia_p = padded_intertias_inv.unsqueeze(-1) * p_reshaped # (N, d+1, 1) * (..., N, d+1, a) = (..., N, d+1, a)
        return (inv_mass_p + inv_intertia_p).reshape(*p.shape)

    def forward(self, t, z):
        self.nfe += 1
        return self.dynamics(t, z)

    def DPhi(self, r, r_dot):
        return rigid_DPhi(self.body_graph, r, r_dot)

    def potential(self, r):
        assert r.ndim == 3
        return self.V_net(r.reshape(r.shape[0], -1))

    def integrate(self, z0, ts, tol=1e-4, method="rk4"):
        """
        input:
            z0: bs, 2, n, d
            ts: length T
        returns:
            a tensor of size bs, T, 2, n, d
        """
        assert (z0.ndim == 4) and (ts.ndim == 1)
        assert z0.shape[-1] == self.d
        assert z0.shape[-2] == self.n
        bs = z0.shape[0]
        zT = odeint(self, z0.reshape(bs, -1), ts, rtol=tol, method=method).permute(1, 0, 2)
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
        