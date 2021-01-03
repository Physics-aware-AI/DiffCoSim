import torch
import torch.nn as nn
from models.dynamics import ConstrainedHamiltonianDynamics, HamiltonianDynamics
from utils import Linear, mlp, Reshape, CosSin
from systems.rigid_body import EuclideanT, GeneralizedT, rigid_DPhi
from torchdiffeq import odeint

class CHNN(nn.Module):
    def __init__(self, body_graph, dof_ndim, num_layers, hidden_size, dtype, **kwargs):
        super().__init__()
        self.body_graph = body_graph
        self.nfe = 0
        self.n = len(self.body_graph.nodes)
        self.d = dof_ndim
        self.q_ndim = self.n * self.d
        self.dynamics = ConstrainedHamiltonianDynamics(self.hamiltonian, self.DPhi)
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

    def M(self, v):
        assert len(self.m_params) == 1 # limited support for now
        *dims, n, a = v.shape
        d = int(list(self.m_params.keys())[0])
        N = n // (d+1) # number of extended bodies
        v_reshaped = v.reshape(*dims, N, d+1, a)
        moments = torch.exp(self.m_params[str(d)])
        masses = moments[:, :1]
        if d == 0:
            return (masses.unsqueeze(-1)*v_reshaped).reshape(*v.shape)
        a00 = (masses + moments[:,1:].sum(-1, keepdims=True)).unsqueeze(-1) # (N, 1, 1)
        ai0 = a0i = -moments[:,1:].unsqueeze(-1) # (N, d, 1)
        p0 = a00 * v_reshaped[..., :1,:] + (a0i * v_reshaped[..., 1:, :]).sum(-2, keepdims=True) # (..., N, 1, a)
        aii = moments[:, 1:].unsqueeze(-1)
        pi = ai0 * v_reshaped[...,:1,:] + aii * v_reshaped[...,1:,:] # (..., N, d, a)
        return torch.cat([p0,pi],dim=-2).reshape(*v.shape)

    def forward(self, t, z):
        self.nfe += 1
        return self.dynamics(t, z)

    def DPhi(self, z):
        bs = z.shape[0]
        r, p = z.reshape(bs, 2, self.n, self.d).unbind(dim=1)
        r_dot = self.Minv(p)
        DPhi = rigid_DPhi(self.body_graph, r, r_dot)
        # convert d/dr_dot to d/dp
        # DPhi[:, 1] = self.Minv(DPhi[:, 1].reshape(bs, self.n, -1)).reshape(DPhi[:,1].shape)
        DPhi = torch.stack([DPhi[:, 0], self.Minv(DPhi[:, 1].reshape(bs, self.n, -1)).reshape(DPhi[:,1].shape)], dim=1)
        return DPhi.reshape(bs, 2*self.n*self.d, -1)

    def hamiltonian(self, t, z):
        assert z.shape[-1] == 2 * self.d *self.n
        bs, D = z.shape
        r = z[:, : D//2].reshape(bs, self.n, -1)
        p = z[:, D//2 :].reshape(bs, self.n, -1)
        T = EuclideanT(p, self.Minv)
        V = self.potential(r)
        return T + V

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
        r0, r_dot0 = z0.chunk(2, dim=1)
        p0 = self.M(r_dot0)

        rp0 = torch.cat([r0, p0], dim=1).reshape(bs, -1) # assume C-style stride
        rpT = odeint(self, rp0, ts, rtol=tol, method=method)
        rpT = rpT.permute(1,0,2) # bs, T, 2*n*d
        rpT = rpT.reshape(bs, len(ts), 2, self.n, self.d)
        rT, pT = rpT.chunk(2, dim=2)
        r_dotT = self.Minv(pT)
        r_r_dotT = torch.cat([rT, r_dotT], dim=2)
        return r_r_dotT


class HNN_Struct(nn.Module):
    def __init__(self, body_graph, dof_ndim, num_layers, hidden_size, dtype, 
                    canonical=False, **kwargs):
        super().__init__()
        # self.body_graph = body_graph
        self.nfe = 0
        self.q_ndim = dof_ndim
        self.canonical = canonical
        self.dynamics = HamiltonianDynamics(self.hamiltonian)
        sizes = [self.q_ndim] + num_layers * [hidden_size] + [1]
        self.V_net = nn.Sequential(
            mlp(sizes, nn.Softplus, orthogonal_init=True), 
            Reshape(-1)
        )
        sizes = [self.q_ndim] + num_layers * [hidden_size] + [self.q_ndim*self.q_ndim]
        self.M_net = nn.Sequential(
            mlp(sizes, nn.Softplus, orthogonal_init=True), 
            Reshape(-1, self.q_ndim, self.q_ndim)
        )

    def tril_Minv(self, q):
        M_q = self.M_net(q)
        res = torch.tril(M_q, diagonal=1)
        # constrain diagonal to be positive
        # res = res + torch.diag_embed(
        #     torch.nn.functional.softplus(torch.diagonal(M_q, dim1=-2, dim2=-1)),
        #     dim1=-2, dim2=-1
        # )
        res = res + torch.diag_embed(
            torch.diagonal(M_q, dim1=-2, dim2=-1),
            dim1=-2, dim2=-1
        )
        return res

    def Minv(self, q, eps=1e-1):
        assert q.ndim == 2
        L_q = self.tril_Minv(q)
        assert L_q.ndim == 3
        diag_noise = eps * torch.eye(L_q.size(-1), dtype=q.dtype, device=q.device)
        Minv = L_q @ L_q.transpose(-2, -1) + diag_noise
        return Minv

    def M(self, q, eps=1e-1):
        assert q.ndim == 2
        L_q = self.tril_Minv(q)
        assert L_q.ndim == 3

        def M_func(qdot):
            assert qdot.ndim == 2
            qdot = qdot.unsqueeze(-1)
            diag_noise = eps * torch.eye(L_q.size(-1), dtype=qdot.dtype, device=qdot.device)
            M_qdot = torch.solve(
                qdot, 
                L_q @ L_q.transpose(-2, -1) + diag_noise
            )[0].squeeze(-1)
            return M_qdot

        return M_func

    def hamiltonian(self, t, z):
        q, p = z.chunk(2, dim=1)
        V = self.V_net(q)
        Minv = self.Minv(q)
        T = GeneralizedT(p, Minv)
        return T + V

    def forward(self, t, z):
        assert (t.ndim == 0) and (z.ndim == 2)
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
        z0 = z0.reshape(bs, -1)
        if self.canonical:
            q0, p0 = z0.chunk(2, dim=1)
        else:
            q0, q_dot0 = z0.chunk(2, dim=1)
            p0 = self.M(q0)(q_dot0)
        qp0 = torch.cat([q0, p0], dim=-1)
        qpT = odeint(self, qp0, ts, rtol=tol, method=method)
        qpT = qpT.permute(1, 0, 2)

        if self.canonical:
            qpT = qpT.reshape(bs, len(ts), 2, D)
            return qpT
        else:
            qT, pT = qpT.reshape(bs*len(ts), 2*D).chunk(2, dim=-1)
            q_dotT = (self.Minv(qT) @ pT.unsqueeze(-1)).squeeze(-1)
            q_q_dotT = torch.cat([qT, q_dotT], dim=-1).reshape(bs, len(ts), 2, D)
            return q_q_dotT


class HNN_Struct_Angle(HNN_Struct):
    def __init__(self, body_graph, dof_ndim, num_layers, 
                    hidden_size, dtype, angular_dims, canonical=False, **kwargs):
        super().__init__(body_graph, dof_ndim, num_layers, hidden_size, dtype, canonical, **kwargs)

        sizes = [self.q_ndim + len(angular_dims)] + num_layers * [hidden_size] + [1]
        self.V_net = nn.Sequential(
            CosSin(self.q_ndim, angular_dims, only_q=True),
            mlp(sizes, nn.Softplus, orthogonal_init=True),
            Reshape(-1)
        )
        sizes = [self.q_ndim + len(angular_dims)] + num_layers * [hidden_size] + [self.q_ndim*self.q_ndim]
        self.M_net = nn.Sequential(
            CosSin(self.q_ndim, angular_dims, only_q=True),
            mlp(sizes, nn.Softplus, orthogonal_init=True),
            Reshape(-1, self.q_ndim, self.q_ndim)
        )

class HNN(HNN_Struct):
    def __init__(self, body_graph, dof_ndim, num_layers, hidden_size, dtype, 
                    canonical=False, **kwargs):
        super().__init__(body_graph, dof_ndim, num_layers, hidden_size, dtype, 
                    canonical=False, **kwargs)
        self.V_net = None
        sizes = [self.q_ndim*2] + num_layers * [hidden_size] + [1]
        self.H_net = nn.Sequential(
            mlp(sizes, nn.Softplus, orthogonal_init=True), 
            Reshape(-1)
        )

    def hamiltonian(self, t, z):
        return self.H_net(z)


class HNN_Angle(HNN):
    def __init__(self, body_graph, dof_ndim, num_layers, hidden_size, dtype, 
                    angular_dims, canonical=False, **kwargs):
        super().__init__(body_graph, dof_ndim, num_layers, hidden_size, dtype, 
                    canonical=False, **kwargs)
        
        sizes = [self.q_ndim*2 + len(angular_dims)] + num_layers * [hidden_size] + [1]
        self.H_net = nn.Sequential(
            CosSin(self.q_ndim, angular_dims, only_q=False),
            mlp(sizes, nn.Softplus, orthogonal_init=True), 
            Reshape(-1)
        )
        sizes = [self.q_ndim + len(angular_dims)] + num_layers * [hidden_size] + [self.q_ndim*self.q_ndim]
        self.M_net = nn.Sequential(
            CosSin(self.q_ndim, angular_dims, only_q=True),
            mlp(sizes, nn.Softplus, orthogonal_init=True), 
            Reshape(-1, self.q_ndim, self.q_ndim)
        )