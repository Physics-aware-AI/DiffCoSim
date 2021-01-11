import torch
import networkx as nx
import numpy as np
from collections import OrderedDict, defaultdict
from models.dynamics import ConstrainedLagrangianDynamics

from pytorch_lightning import seed_everything
from torchdiffeq import odeint
from utils import Animation

class BodyGraph(nx.Graph):
    def __init__(self):
        super().__init__()
        self.key2id = OrderedDict()
        self.d2ids = defaultdict(list)

    def add_node(self, key, *args, **kwargs):
        self.key2id[key] = len(self.key2id)
        super().add_node(key, *args, **kwargs)

    def add_extended_body(self, key, m, moments=None, d=0, **kwargs):
        self.add_node(key, m=m, d=d, **kwargs)
        self.d2ids[d].extend([self.key2id[key]+i for i in range(d+1)])
        for i in range(d):
            child_key = f'{key}_{i}'
            self.add_node(child_key)
            self.add_edge(key, child_key, internal=True, l=1., I=m*moments[i])
            for j in range(i):
                self.add_edge(f'{key}_{j}', child_key, internal=True, l=np.sqrt(2))
    
    def add_joint(self, key1, pos1, key2=None, pos2=None, rotation_axis=None):
        """ adds a joint between extended bodies key1 and key2 at the position
            in the body frame 1 pos1 and body frame 2 pos2. pos1 and pos2 should
            be d dimensional vectors, where d is the dimension of the extended body.
            If key2 is not specified, the joint connection is to a fixed point in space pos2."""
        if key2 is not None:
            if rotation_axis is None:
                self.add_edge(key1, key2, external=True, joint=(pos1, pos2))
            else:
                self.add_edge(key1, key2, external=True, joint=(pos1, pos2), rotation_axis=rotation_axis)
        else:
            self.nodes[key1]['joint'] = (pos1, pos2)
            if rotation_axis is not None:
                self.nodes[key1]['rotation_axis'] = rotation_axis

class RigidBody(object):
    dt = 0.1
    integration_time = 10.0
    _m, _minv = None, None

    def mass_matrix(self):
        # n = len(self.body_graph.nodes)
        n = self.n
        M = torch.zeros(n, n, dtype=torch.float64)
        for key, mass in nx.get_node_attributes(self.body_graph, "m").items():
            id = self.body_graph.key2id[key]
            M[id, id] += mass
        for (key_i, key_j), I in nx.get_edge_attributes(self.body_graph, "I").items():
            id_i, id_j = self.body_graph.key2id[key_i], self.body_graph.key2id[key_j]
            M[id_i, id_i] += I
            M[id_i, id_j] -= I
            M[id_j, id_i] -= I
            M[id_j, id_j] += I
        return M

    @property
    def M(self):
        if self._m is None:
            self._m = self.mass_matrix()
        return self._m
    
    @property
    def Minv(self):
        if self._minv is None:
            self._minv = self.M.inverse()
        return self._minv

    def Minv_op(self, p):
        # p: (*dims, n, a)
        return self.Minv.to(p.device, p.dtype) @ p
    
    def DPhi(self, r, v):
        return rigid_DPhi(self.body_graph, r, v)

    def potential(self, r):
        raise NotImplementedError

    # def hamiltonian(self, t, z):
    #     bs, D = z.shape
    #     r = z[:, : D//2].reshape(bs, self.n, -1)
    #     p = z[:, D//2 :].reshape(bs, self.n, -1)
    #     T = EuclideanT(p, self.Minv)
    #     V = self.potential(r)
    #     return T + V

    def dynamics(self):
        return ConstrainedLagrangianDynamics(self.potential, self.Minv_op, self.DPhi, (self.n, self.d))

    # def integrate(self, z0, T, tol=1e-7, method="rk4"):
    #     bs = z0.shape[0]
    #     M = self.M.to(dtype=z0.dtype, device=z0.device)
    #     Minv = self.Minv.to(dtype=z0.dtype, device=z0.device)
    #     rp = torch.stack(
    #         [z0[:,0], M @ z0[:, 1]], dim=1
    #     ).reshape(bs, -1)
    #     with torch.no_grad():
    #         rpT = odeint(self.dynamics(), rp, T.to(dtype=z0.dtype, device=z0.device), rtol=tol, method=method)
    #     rps = rpT.permute(1,0,2).reshape(bs, len(T), *z0.shape[1:])
    #     rvs = torch.stack([rps[:,:,0], Minv @ rps[:,:,1]], dim=2)
    #     return rvs

    def integrate(self, xv0, T, tol=1e-7, method="rk4"):
        bs = xv0.shape[0]
        mus = self.mus.to(xv0.device, xv0.dtype)
        cors = self.cors.to(xv0.device, xv0.dtype)
        Minv = self.Minv.to(xv0.device, xv0.dtype)
        with torch.no_grad():
            xvt = xv0.reshape(bs, -1)
            xvT = torch.zeros([bs, len(T), xvt.shape[1]], device=xvt.device, dtype=xvt.dtype)
            xvT[:, 0] = xvt
            T = T.to(xv0.device, xv0.dtype)
            is_cld_T = torch.zeros(bs, len(T), device=xvt.device, dtype=torch.bool)
            for i in range(len(T)-1):
                xvt_n = odeint(self.dynamics(), xvt, T[i:i+2].to(xv0.device, xv0.dtype), rtol=tol, method=method)[1]
                xvt_n, is_cld = self.impulse_solver.add_impulse(xvt_n, mus, cors, Minv)
                xvt = xvt_n
                xvT[:, i+1] = xvt
                is_cld_T[:, i+1] = is_cld
        return xvT.reshape(bs, len(T), *xv0.shape[1:]), is_cld_T

    def animate(self, zt, j=None):
        # bs, T, 2, n, d
        if zt.ndim == 5:
            j = j if j is not None else np.random.randint(zt.shape[0])
            traj = zt[j, :, 0, :, :]
        else:
            traj = zt[:, 0, :, :]
        anim = self.animator(traj, self)
        return anim.animate()

    @property
    def animator(self):
        return Animation


def rigid_DPhi(G, r, v):
    constraints = (dist_constraints_DPhi, joint_constraints_DPhi)
    DPhi = torch.cat([constr(G, r, v) for constr in constraints], dim=-1)
    return DPhi

def joint_constraints_DPhi(G, r, v):
    """ r: bs, n, d
        v: bs, n, d

        outputs: dPhi/d[r,v]: bs, 2, n, d, 2, C
        dim 1: 0: /dr, 1: /dv
        dim -2: 0: dphi, 1: d\dot_phi
    """
    bs, n, d = r.shape
    edge_joints = nx.get_edge_attributes(G, 'joint')
    node_joints = nx.get_node_attributes(G, 'joint')
    disabled_axes = nx.get_edge_attributes(G, 'rotation_axis')
    num_constraints = len(edge_joints) + len(node_joints) + len(disabled_axes)
    # each vector constraint <-> d scalar constraints
    DPhi = torch.zeros(bs, 2, n, d, 2, num_constraints, d, device=r.device, dtype=r.dtype)
    # joints connecting two bodies
    jid = 0
    for ((key_i, key_j), (c1, c2)) in edge_joints.items():
        id_i, id_j = G.key2id[key_i], G.key2id[key_j]
        c1t = torch.cat([1-c1.sum()[None], c1]).to(r.device, r.dtype)
        c2t = torch.cat([1-c2.sum()[None], c2]).to(r.device, r.dtype)
        di = G.nodes[key_i]['d']
        dj = G.nodes[key_j]['d']
        for k in range(d): # (bs, 4, d, d) 
            # dphi/dr
            DPhi[:,0,id_i:id_i+1+di,k,0,jid,k] = c1t
            DPhi[:,0,id_j:id_j+1+dj,k,0,jid,k] = -c2t
            # dphi/dv always zero
            # dphi_dot/dr always zero
            # dphi_dot/dv
            DPhi[:,1,id_i:id_i+1+di,k,1,jid,k] = c1t
            DPhi[:,1,id_j:id_j+1+dj,k,1,jid,k] = -c2t
        jid += 1
        if 'rotation_axis' in G[key_i][key_j]:
            raise NotImplementedError # Todo

    # joints connecting a body to a fixed point in space
    for jid2, (key_i, (c1, _)) in enumerate(node_joints.items()):
        id_i = G.key2id[key_i]
        c1t = torch.cat([1-c1.sum()[None], c1]).to(r.device, r.dtype)
        di = G.nodes[key_i]["d"]
        for k in range(d): # (bs, di+1, d, d)
            DPhi[:,0,id_i:id_i+1+di,k,0,jid+jid2,k] = c1t
            DPhi[:,1,id_i:id_i+1+di,k,1,jid+jid2,k] = c1t
    
    return DPhi.reshape(bs,2,n,d,2,-1)

def dist_constraints_DPhi(G, r, v):
    """ r: bs, n, d
        v: bs, n, d

        outputs: dPhi/d [r, v]: bs, 2, n, d, 2, C
        dim 1: 0: /dr, 1: /dv
        dim -2: 0: dphi, 1: d\dot_phi
    """
    bs, n, d = r.shape
    p2p_constrs = nx.get_edge_attributes(G, 'l'); p2ps = len(p2p_constrs)
    # tethers
    tether_constrs = nx.get_node_attributes(G, "tether"); tethers = len(tether_constrs)
    DPhi = torch.zeros(bs, 2, n, d, 2, p2ps+tethers, device=r.device, dtype=r.dtype)
    # Fixed distance between two points
    for cid, ((key_i, key_j), _) in enumerate(p2p_constrs.items()):
        id_i, id_j = G.key2id[key_i], G.key2id[key_j]
        # dphi/dr
        DPhi[:, 0, id_i, :, 0, cid] = 2 * (r[:, id_i] - r[:, id_j])
        DPhi[:, 0, id_j, :, 0, cid] = 2 * (r[:, id_j] - r[:, id_i])
        # dphi/dr_dot is always zero
        # Dphi [:, 1, id_i, :, 0, cid]
        # d\phi_dot/dr
        DPhi[:, 0, id_i, :, 1, cid] = 2 * (v[:, id_i] - v[:, id_j])
        DPhi[:, 0, id_j, :, 1, cid] = 2 * (v[:, id_j] - v[:, id_i])
        # d\phi_dot/dr_dot
        DPhi[:, 1, id_i, :, 1, cid] = 2 * (r[:, id_i] - r[:, id_j])
        DPhi[:, 1, id_j, :, 1, cid] = 2 * (r[:, id_j] - r[:, id_i])
    # Fixed distance between a point and a fixed point in space
    for cid, (key_i, (pos, _)) in enumerate(tether_constrs.items()):
        id_i = G.key2id[key_i]
        r0 = pos.to(r.device, r.dtype)
        DPhi[:, 0, id_i, :, 0, p2ps+cid] = 2 * (r[:, id_i] - r0)
        DPhi[:, 0, id_i, :, 1, p2ps+cid] = 2 * v[:, id_i]
        DPhi[:, 1, id_i, :, 1, p2ps+cid] = 2 * (r[:, id_i] - r0) 
    
    return DPhi



def EuclideanT(p, Minv):
    """ p^T Minv p / 2
    p: bs, ndof, D
    Minv: (bs, ndof, ndof) or (ndof, ndof)

    """
    assert p.ndim == 3
    Minv_p = Minv(p) if callable(Minv) else Minv.to(dtype=p.dtype, device=p.device).matmul(p) 
    assert Minv_p.ndim == 3
    return (p * Minv_p).sum((-1, -2)) / 2.0

def GeneralizedT(p, Minv):
    """
    p: bs, D
    Minv: bs, D, D or D * D
    """
    assert p.ndim == 2
    Minv_p = Minv(p) if callable(Minv) else Minv.to(dtype=p.dtype, device=p.device).matmul(p.unsqueeze(-1)).squeeze(-1)
    assert Minv_p.ndim == 2
    return (p * Minv_p).sum((-1,)) / 2.0

