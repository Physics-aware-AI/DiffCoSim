import torch
import networkx as nx
import numpy as np
from collections import OrderedDict, defaultdict
from models.dynamics import ConstrainedLagrangianDynamics

from pytorch_lightning import seed_everything
from torchdiffeq import odeint
from utils import Animation

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

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
        M = torch.zeros(n, n, dtype=torch.float32)
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
        with torch.no_grad():
            xvt = xv0.reshape(bs, -1)
            xvT = torch.zeros([bs, len(T), xvt.shape[1]], device=xvt.device, dtype=xvt.dtype)
            xvT[:, 0] = xvt
            for i in range(len(T)-1):
                xvt_n = odeint(self.dynamics(), xvt, T.to(xv0.device, xv0.dtype), rtol=tol, method=method)[1]
                xvt_n = self.add_impulse(xvt_n)
                xvt = xvt_n
                xvT[:, i+1] = xvt
        return xvT.reshape(bs, len(T), *xv0.shape[1:]) 

    def add_impulse(self, xv):
        bs = xv.shape[0]
        n, d = self.n, self.d
        x, v = xv.reshape(bs, 2, n, d).unbind(dim=1)
        is_cld, is_cld_ij, is_cld_bdry, dist_ij, dist_bdry = self.check_collision(x)
        if is_cld.sum() == 0:
            return xv

        # deal with collision individually. 
        for bs_idx in torch.nonzero(is_cld).squeeze(1):
            # construct contact Jacobian
            cld_ij_ids = torch.nonzero(is_cld_ij[bs_idx]) # n_cld_ij, 2
            cld_bdry_ids = torch.nonzero(is_cld_bdry[bs_idx]) # n_cld_bdry, 2

            n_cld_ij = len(cld_ij_ids) ; n_cld_bdry = len(cld_bdry_ids)
            n_cld = n_cld_ij + n_cld_bdry

            # contact Jacobian (n_cld, d, n, d)
            Jac = self.get_contact_Jacobian(bs_idx, x, cld_ij_ids, cld_bdry_ids)
            # Jac_v, v_star, mu, cor
            Jac_v = Jac.reshape(n_cld*d, n*d) @ v[bs_idx].reshape(n*d, 1) # n_cld*d, 1
            v_star = torch.zeros([n_cld, d], dtype=Jac_v.dtype, device=Jac_v.device)
            v_star[:,0] = torch.cat([
                - dist_ij[bs_idx, cld_ij_ids[:,0], cld_ij_ids[:,1]] / self.dt / 8,
                - dist_bdry[bs_idx, cld_bdry_ids[:,0], cld_bdry_ids[:,1]] / self.dt / 8,
            ], dim=0) # n_cld, d
            
            mu = self.mus[self.cld_2did_to_1did(cld_ij_ids, cld_bdry_ids)].to(x.device, x.dtype)
            cor = self.cors[self.cld_2did_to_1did(cld_ij_ids, cld_bdry_ids)]

            # get equality constraints
            DPhi = self.DPhi(x[bs_idx:bs_idx+1], v[bs_idx:bs_idx+1]) # 

            if DPhi.shape[-1] != 0: # exist equality constraints
                self.update_v_w_J_e(bs_idx, v, Jac, Jac_v, v_star, mu, cor, DPhi)
            else:                   # no equality constraints
                self.update_v_wo_J_e(bs_idx, v, Jac, Jac_v, v_star, mu, cor)

        return torch.stack([x, v], dim=1).reshape(bs, -1)

    def get_contact_Jacobian(self, bs_idx, x, cld_ij_ids, cld_bdry_ids):
        bs, n_o, n_p, d = x.shape[0], self.n_o, self.n_p, self.d
        n_cld_ij = len(cld_ij_ids) ; n_cld_bdry = len(cld_bdry_ids)
        n_cld = n_cld_ij + n_cld_bdry

        x = x.reshape(bs, n_o, n_p, d)
        # calculate cld_bdry contact point coordinates
        o1_bdry = x[bs_idx, cld_bdry_ids[:, 0]] # n_cld_bdry, n_p, d
        p1_bdry = o1_bdry[..., 0, :] # n_cld_bdry, d
        # p1_bdry = x[bs_idx, cld_bdry_ids[:, 0], 0] # n_cld_bdry, d
        a, b, c = self.bdry_lin_coef[cld_bdry_ids[:, 1]].unbind(dim=1) # n_cld_bdry, 3 -> n_cld_bdry,
        pc_bdry = torch.stack(
            [ b*b*p1_bdry[:,0] - a*b*p1_bdry[:,1] -a*c, a*a*p1_bdry[:,1] - a*b*p1_bdry[:,0] + b*c],
            dim=1,
        ) / (a*a + b*b).unsqueeze(dim=1)

        # cld_ij local frame: e_n_ij, e_t_ij
        o1_ij = x[bs_idx, cld_ij_ids[:, 0]] # n_cld_ij, n_p, d
        o2_ij = x[bs_idx, cld_ij_ids[:, 1]] # n_cld_ij, n_p, d
        p1_ij = o1_ij[..., 0, :] # n_cld_ij, d
        p2_ij = o2_ij[..., 0, :] # n_cld_ij, d
        # p1_ij, p2_ij = x[bs_idx, cld_ij_ids, 0].unbind(dim=1) # n_cld_ij, 2, d -> n_cld_ij, d

        e_n_ij, e_t_ij = self.get_contact_coordinate_frame(p1_ij, p2_ij)
        e_n_bdry, e_t_bdry = self.get_contact_coordinate_frame(p1_bdry, pc_bdry)

        ls1_ij = self.ls[cld_ij_ids[:, 0]] # (n_cld_ij,)
        ls2_ij = self.ls[cld_ij_ids[:, 1]]
        ls1_bdry = self.ls[cld_bdry_ids[:, 0]]
        c_til_ij_1 = self.get_c_tilde(o1_ij, e_n_ij, ls1_ij)
        c_til_ij_2 = self.get_c_tilde(o2_ij, - e_n_ij, ls2_ij)
        c_til_bdry_1 = self.get_c_tilde(o1_bdry, e_n_bdry, ls1_bdry)

        Jac = torch.zeros([n_cld, d, n_o, n_p, d], device=x.device, dtype=x.dtype)
        for cid, (i, j) in enumerate(cld_ij_ids):
            Jac[cid, 0, i] = - c_til_ij_1[cid, :, None] * e_n_ij[cid, None, :] # n_p, d
            Jac[cid, 0, j, :] = c_til_ij_2[cid, :, None] * e_n_ij[cid, None, :] # n_p, d
            Jac[cid, 1, i, :] = - c_til_ij_1[cid, :, None] * e_t_ij[cid, None, :] # n_p, d
            Jac[cid, 1, j, :] = c_til_ij_2[cid, :, None] * e_t_ij[cid, None, :] # n_p, d
        
        for cid, (i, _) in enumerate(cld_bdry_ids):
            Jac[n_cld_ij+cid, 0, i, :] = - c_til_bdry_1[cid, :, None] * e_n_bdry[cid, None, :]
            Jac[n_cld_ij+cid, 1, i, :] = - c_til_bdry_1[cid, :, None] * e_t_bdry[cid, None, :]

        return Jac

    def get_contact_coordinate_frame(self, p1, p2):
        # local frame in contact space, e_n is from obj1 to obj2
        e_n = (p2 - p1) / ((p2 - p1) ** 2).sum(1, keepdim=True).sqrt() # n_cld, d
        e_t = torch.zeros_like(e_n) # n_cld, 2
        e_t[..., 0], e_t[..., 1] = -e_n[..., 1], e_n[..., 0]
        return e_n, e_t

    def get_c_tilde(self, obj, e_n, ls):
        """ inputs 
                obj: (*N, n_p, d)
                e: (*N, d)
                ls: (*N,)
            outputs:
                c_tilte: (*N, n_p)
        """
        # collision point in local coordinate
        if self.n_p == 1:
            return torch.ones(*ls.shape, 1, dtype=ls.dtype, device=ls.device)
        delta = self.delta.to(obj.device, obj.dtype)
        c_cld = (e_n[..., None, :] * (delta @ obj)).sum(-1) * ls[:, None] # (*N, d)
        c_tilde = torch.cat([
            1-c_cld.sum(-1, keepdim=True), c_cld
        ], dim=-1)
        return c_tilde

    def update_v_w_J_e(self, bs_idx, v, Jac, Jac_v, v_star, mu, cor, DPhi):
        n_cld, d, n_o, n_p, _ = Jac.shape
        n = n_o*n_p
        C = DPhi.shape[-1]
        Minv = self.Minv.to(v.device, v.dtype)
        # calculate A_decom
        J_e_T = DPhi[:,0,:,:,0,:].reshape(n*d, C)
        J_e = J_e_T.t() # 
        Minv_J_e_T = (Minv @ J_e_T.reshape(n, d*C)).reshape(n*d, C)
        J_e_Minv_J_e_T = J_e @ Minv_J_e_T # (C, C)
        M_e = torch.inverse(J_e_Minv_J_e_T) # (C, C) inertia in equality contact space
        L = torch.cholesky(Minv) # (n, n)
        L_T = L.t() # (n, n)
        L_T_J_e_T = (L_T @ J_e_T.reshape(n, d*C)).reshape(n*d, C)
        Q = (L_T_J_e_T @ M_e) @ L_T_J_e_T.t() # (n*d, n*d)
        e, V = torch.symeig(Q, eigenvectors=True)
        assert torch.allclose(e, (e>0.5).to(dtype=e.dtype), atol=1e-6)
        L_V_s = (L @ V[:, :n*d-C].reshape(n, d*(n*d-C))).reshape(n*d, n*d-C) # (n*d, n*d-C)
        V_s_T_L_T_Jac_T = L_V_s.t() @ Jac.reshape(n_cld*d, n*d).t() # (n*d-C, n_cld*d)
        A_decom = V_s_T_L_T_Jac_T
        # compression phase impulse
        impulse, impulse_star = self.get_impulse(A_decom, Jac_v, v_star, mu, n_cld, d, target_impulse=None)
        # velocity after compression phase (before retitution phase)
        M_hat_inv = L_V_s @ L_V_s.t() #(n*d, n*d)
        M_hat_inv_Jac_T = M_hat_inv @ Jac.reshape(n_cld*d, n*d).t() # (n*d, n_cld*d)
        v_prev_r = v[bs_idx] + (M_hat_inv_Jac_T @ impulse_star).reshape(n, d)
        # restitution phase impulse
        Jac_v_prev_r = Jac.reshape(n_cld*d, n*d) @ v_prev_r.reshape(n*d, 1) # n_cld*d, 1
        target_impulse = (cor.reshape(-1, 1) * impulse.reshape(n_cld, d)).reshape(n_cld*d, 1)
        impulse_r, impulse_star_r = self.get_impulse(A_decom, Jac_v_prev_r, v_star, mu, n_cld, d, target_impulse=target_impulse)
        # velocity after restitution phase
        v[bs_idx] = v_prev_r + (M_hat_inv_Jac_T @ impulse_r).reshape(n, d)

    def update_v_wo_J_e(self, bs_idx, v, Jac, Jac_v, v_star, mu, cor):
        n_cld, d, n_o, n_p, _ = Jac.shape
        n = n_o * n_p
        Minv = self.Minv.to(v.device, v.dtype)
        # calculate A_decom
        Minv_sqrt = torch.cholesky(Minv) # (n, n)
        Minv_sqrt_Jac_T = (Minv_sqrt @ Jac.reshape(n_cld*d, n*d).t().reshape(n, d*n_cld*d)).reshape(n*d, n_cld*d)
        A_decom = Minv_sqrt_Jac_T
        # compression phase impulse
        impulse, impulse_star = self.get_impulse(A_decom, Jac_v, v_star, mu, n_cld, d, target_impulse=None)
        # velocity after compression phase (before retitution phase)
        Minv_Jac_T = (Minv @ Jac.reshape(n_cld*d, n*d).t().reshape(n, d*n_cld*d)).reshape(n*d, n_cld*d)
        v_prev_r = v[bs_idx] + (Minv_Jac_T @ impulse_star).reshape(n, d)
        # restitution phase impulse
        Jac_v_prev_r = Jac.reshape(n_cld*d, n*d) @ v_prev_r.reshape(n*d, 1) # n_cld*d, 1
        target_impulse = (cor.reshape(-1, 1) * impulse.reshape(n_cld, d)).reshape(n_cld*d, 1)
        impulse_r, impulse_star_r = self.get_impulse(A_decom, Jac_v_prev_r, v_star, mu, n_cld, d, target_impulse=target_impulse)
        # velocity after restitution phase
        v[bs_idx] = v_prev_r + (Minv_Jac_T @ impulse_r).reshape(n, d)

    def get_impulse(self, A_decom, v_, v_star, mu, n_cld, d, target_impulse):
        """
        collision phase: target_impulse = None
        restitution phase: target_impulse = COR * impulse_from_collision_phase
        """
        # v_: (n_cld*d, 1)
        n_cld_d = v_.shape[0]
        f = cp.Variable((n_cld_d, 1))
        A_decom_p = cp.Parameter(A_decom.shape) # Todo
        v_p = cp.Parameter((n_cld_d, 1))
        v_star_p = cp.Parameter((n_cld_d, 1))
        mu_p = cp.Parameter((mu.shape[0], 1)) 

        objective = cp.Minimize(0.5 * cp.sum_squares(A_decom_p @ f) + cp.sum(cp.multiply(f, v_p)) - cp.sum(cp.multiply(f, v_star_p)))
        constraints = [cp.SOC(cp.multiply(mu_p[i], f[i*d]), f[i*d+1:i*d+d]) for i in range(n_cld)]
        if target_impulse is None:
            constraints = constraints + [f[i*d] >= 0 for i in range(n_cld)]
        else:
            constraints = constraints + [f[i*d] >= target_impulse[i*d] for i in range(n_cld)]

        problem = cp.Problem(objective, constraints)
        cvxpylayer = CvxpyLayer(problem, parameters=[A_decom_p, v_p, v_star_p, mu_p], variables=[f])

        impulse, = cvxpylayer(A_decom, v_.reshape(-1, 1), torch.zeros_like(v_star.reshape(-1, 1)), mu.reshape(-1, 1))
        impulse_star, = cvxpylayer(A_decom, v_.reshape(-1, 1), v_star.reshape(-1, 1), mu.reshape(-1, 1))
        return impulse, impulse_star

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

