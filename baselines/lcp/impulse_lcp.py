import fsspec, os
import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from diffcp.cone_program import SolverError
# from symeig import symeig

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .lcp import LCPFunction
lcp_solver = LCPFunction.apply

class ImpulseSolverLCP(nn.Module):
    def __init__(
        self, dt, n_o, n_p, d, 
        check_collision, cld_2did_to_1did, DPhi,  
        ls, bdry_lin_coef, delta=None, 
        get_limit_e_for_Jac=None,
        get_3d_contact_point_c_tilde=None,
        save_dir=os.path.join(PARENT_PARENT_DIR, "tensors")
    ):
        super().__init__()
        assert delta is not None or n_p == 1
        self.dt = dt
        self.n_o = n_o
        self.n_p = n_p
        self.n = n_o * n_p
        self.d = d
        self.cld_2did_to_1did = cld_2did_to_1did
        self.check_collision = check_collision
        self.DPhi = DPhi
        self.register_buffer("ls", ls)
        self.register_buffer("bdry_lin_coef", bdry_lin_coef)
        if delta is None:
            self.delta = None
        else:
            self.register_buffer("delta", delta)
        self.get_limit_e_for_Jac = get_limit_e_for_Jac
        self.get_3d_contact_point_c_tilde = get_3d_contact_point_c_tilde
        self.save_dir = save_dir
        

    def add_impulse(self, xv, mus, cors, Minv):
        bs = xv.shape[0]
        n, d = self.n, self.d
        x = xv.reshape(bs, 2, n, d)[:, 0]
        v = xv.reshape(bs, 2, n, d)[:, 1]
        # x, v = xv.reshape(bs, 2, n, d).unbind(dim=1)
        is_cld, is_cld_ij, is_cld_bdry, is_cld_limit, dist_ij, dist_bdry, dist_limit= self.check_collision(x)
        if is_cld.sum() == 0:
            return xv, is_cld
        # for the rope tasks
        if is_cld_limit.shape[1] > 0:
            self.e_n_limit_for_Jac, self.e_t_limit_for_Jac = self.get_limit_e_for_Jac(x)
        # specifically for the gyroscope task
        if self.get_3d_contact_point_c_tilde is not None:
            self.contact_c_tilde = self.get_3d_contact_point_c_tilde(x)

        # deal with collision individually. 
        new_v = 1 * v
        for bs_idx in torch.nonzero(is_cld, as_tuple=False).squeeze(1):
            # construct contact Jacobian
            cld_ij_ids = torch.nonzero(is_cld_ij[bs_idx], as_tuple=False) # n_cld_ij, 2
            cld_bdry_ids = torch.nonzero(is_cld_bdry[bs_idx], as_tuple=False) # n_cld_bdry, 2
            cld_limit_ids = torch.nonzero(is_cld_limit[bs_idx], as_tuple=False) # n_cld_limit, 

            n_cld_ij = len(cld_ij_ids) ; n_cld_bdry = len(cld_bdry_ids)
            n_cld_limit = len(cld_limit_ids)
            n_cld = n_cld_ij + n_cld_bdry + n_cld_limit

            # contact Jacobian (n_cld, d, n, d)
            Jac = self.get_contact_Jacobian(bs_idx, x, cld_ij_ids, cld_bdry_ids, cld_limit_ids)
            # Jac_v, v_star, mu, cor
            Jac_v = Jac.reshape(n_cld*d, n*d) @ v[bs_idx].reshape(n*d, 1) # n_cld*d, 1
            v_star = torch.zeros([n_cld, d], dtype=Jac_v.dtype, device=Jac_v.device)
            v_star[:,0] = torch.cat([
                - dist_ij[bs_idx, cld_ij_ids[:,0], cld_ij_ids[:,1]] / self.dt / 8,
                - dist_bdry[bs_idx, cld_bdry_ids[:,0], cld_bdry_ids[:,1]] / self.dt / 8,
                - dist_limit[bs_idx, cld_limit_ids[:,0], cld_limit_ids[:,1]] / self.dt / 8
            ], dim=0) # n_cld, d

            
            mu = mus[self.cld_2did_to_1did(cld_ij_ids, cld_bdry_ids, cld_limit_ids)]
            cor = cors[self.cld_2did_to_1did(cld_ij_ids, cld_bdry_ids, cld_limit_ids)]

            # get equality constraints
            DPhi = self.DPhi(x[bs_idx:bs_idx+1], v[bs_idx:bs_idx+1]) # 
            if DPhi.shape[-1] != 0: # exist equality constraints
                # dv = self.get_dv_w_J_e(bs_idx, v, Minv, Jac, Jac_v, v_star, mu, cor, DPhi)
                C = DPhi.shape[-1]
                J_e_T = DPhi[:,0,:,:,0,:].reshape(n*d, C)
                J_e = J_e_T.t().unsqueeze(0) # (1, C, n*d)
                b = torch.zeros([1, C]).type_as(x)
            else:                   # no equality constraints
                # dv = self.get_dv_wo_J_e(bs_idx, v, Minv, Jac, Jac_v, v_star, mu, cor)
                J_e = torch.tensor([]).type_as(x)
                b = torch.tensor([]).type_as(x)
            # get contact jacobian (the normal component)
            J_cn = Jac[:, 0].reshape(1, n_cld, n*d)
            # the tangential jacobian takes both opposte directions into account
            J_ct = torch.zeros([1, n_cld*(d-1)*2, n*d])
            if d == 2:
                J_ct[0, 0::2, :] = Jac[:, 1].reshape(n_cld, n*d)
                J_ct[0, 1::2, :] = -Jac[:, 1].reshape(n_cld, n*d)
            else:
                J_ct[0, 0::4] = Jac[:, 1].reshape(n_cld, n*d)
                J_ct[0, 1::4] = Jac[:, 2].reshape(n_cld, n*d)
                J_ct[0, 2::4] = - Jac[:, 1].reshape(n_cld, n*d)
                J_ct[0, 3::4] = - Jac[:, 2].reshape(n_cld, n*d)
            # populate mass matrix to n*d, n*d
            M = torch.inverse(Minv)
            populated_M = torch.zeros([1, n*d, n*d]).type_as(x)
            for i in range(d):
                populated_M[0, i::d, i::d] = M
            
            mu_matrix = torch.diag(mu).unsqueeze(0) # (n_cld, n_cld)

            E = torch.zeros((1, n_cld*(d-1)*2, n_cld)).type_as(x)
            for i in range(n_cld):
                E[0, i*2*(d-1):(i+1)*2*(d-1), i] = torch.ones(2*(d-1)).type_as(E)
            
            Mv_ = (populated_M @ v[bs_idx].reshape(n*d, 1)).squeeze(2) # (1, n*d)

            J_cn_v_ = (J_cn @ v[bs_idx].reshape(n*d, 1)).squeeze(2) # (1, n_cld)

            # prepare matrices for the LCP
            h = torch.cat(
                [torch.min(-v_star[:,0].unsqueeze(0), cor*J_cn_v_), torch.zeros((1, n_cld*(d-1)*2+n_cld)).type_as(x)],
                dim=1
            )#, the first entry must be a negative number, 

            G = torch.cat(
                [J_cn, J_ct, torch.zeros([1, n_cld, n*d])], dim=1
            )

            F = torch.zeros([1, G.size(1), G.size(1)]).type_as(x)
            F[:, -mu.size(0):, :mu.size(0)] = mu_matrix
            F[:, J_cn.size(1):-E.size(2), -E.size(2):] = E
            F[:, -mu.size(0):, mu.size(0):mu.size(0) + E.size(1)] = - E.transpose(1, 2)

            v_plus = -lcp_solver(populated_M, Mv_, G, h, J_e, b, -F)
            dv = torch.zeros_like(new_v)
            dv[bs_idx] = - new_v[bs_idx] + v_plus.reshape(n, d)

            new_v = new_v + dv

        return torch.stack([x, new_v], dim=1).reshape(bs, -1), is_cld

    def get_contact_Jacobian(self, bs_idx, x, cld_ij_ids, cld_bdry_ids, cld_limit_ids):
        bs, n_o, n_p, d = x.shape[0], self.n_o, self.n_p, self.d
        n_cld_ij = len(cld_ij_ids) ; n_cld_bdry = len(cld_bdry_ids)
        n_cld_limit = len(cld_limit_ids)
        n_cld = n_cld_ij + n_cld_bdry + n_cld_limit

        x = x.reshape(bs, n_o, n_p, d)
        if d == 2:
            # calculate cld_bdry contact point coordinates
            o1_bdry = x[bs_idx, cld_bdry_ids[:, 0]] # n_cld_bdry, n_p, d
            p1_bdry = o1_bdry[..., 0, :] # n_cld_bdry, d
            # p1_bdry = x[bs_idx, cld_bdry_ids[:, 0], 0] # n_cld_bdry, d
            a, b, c = self.bdry_lin_coef[cld_bdry_ids[:, 1]].unbind(dim=1) # n_cld_bdry, 3 -> n_cld_bdry,
            pc_bdry = torch.stack(
                [ b*b*p1_bdry[:,0] - a*b*p1_bdry[:,1] -a*c, a*a*p1_bdry[:,1] - a*b*p1_bdry[:,0] - b*c],
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
            # limited support for the constraint type "limit"
            for cid, (i_limit, t_limit) in enumerate(cld_limit_ids):
                if i_limit < n_o - 1: # bending constraint
                    if t_limit == 0: # angle of the latter link is large
                        if i_limit >= 1:
                            Jac[n_cld_ij+n_cld_bdry+cid, 0, i_limit-1] = - self.e_n_limit_for_Jac[bs_idx, i_limit, None, :]
                        Jac[n_cld_ij+n_cld_bdry+cid, 0, i_limit] = self.e_n_limit_for_Jac[bs_idx, i_limit, None, :] + self.e_n_limit_for_Jac[bs_idx, i_limit+1, None, :]
                        Jac[n_cld_ij+n_cld_bdry+cid, 0, i_limit+1] = - self.e_n_limit_for_Jac[bs_idx, i_limit+1, None, :]
                        if i_limit >= 1:
                            Jac[n_cld_ij+n_cld_bdry+cid, 1, i_limit-1] = - self.e_t_limit_for_Jac[bs_idx, i_limit, None, :]
                        Jac[n_cld_ij+n_cld_bdry+cid, 1, i_limit] = self.e_t_limit_for_Jac[bs_idx, i_limit, None, :] + self.e_t_limit_for_Jac[bs_idx, i_limit+1, None, :]
                        Jac[n_cld_ij+n_cld_bdry+cid, 1, i_limit+1] = - self.e_t_limit_for_Jac[bs_idx, i_limit+1, None, :]
                    else: # angle fo the former link is large
                        if i_limit >= 1:
                            Jac[n_cld_ij+n_cld_bdry+cid, 0, i_limit-1] = self.e_n_limit_for_Jac[bs_idx, i_limit, None, :]
                        Jac[n_cld_ij+n_cld_bdry+cid, 0, i_limit] = - self.e_n_limit_for_Jac[bs_idx, i_limit, None, :] - self.e_n_limit_for_Jac[bs_idx, i_limit+1, None, :]
                        Jac[n_cld_ij+n_cld_bdry+cid, 0, i_limit+1] = self.e_n_limit_for_Jac[bs_idx, i_limit+1, None, :]
                        if i_limit >= 1:
                            Jac[n_cld_ij+n_cld_bdry+cid, 1, i_limit-1] = self.e_t_limit_for_Jac[bs_idx, i_limit, None, :]
                        Jac[n_cld_ij+n_cld_bdry+cid, 1, i_limit] = - self.e_t_limit_for_Jac[bs_idx, i_limit, None, :] - self.e_t_limit_for_Jac[bs_idx, i_limit+1, None, :]
                        Jac[n_cld_ij+n_cld_bdry+cid, 1, i_limit+1] = self.e_t_limit_for_Jac[bs_idx, i_limit+1, None, :]
                else: # stretching constraint
                    i_limit_offset = i_limit - (n_o-1)
                    if t_limit == 0: # maximum stretch
                        if i_limit_offset >= 1:
                            Jac[n_cld_ij+n_cld_bdry+cid, 0, i_limit_offset-1] = self.e_n_limit_for_Jac[bs_idx, n_o+i_limit_offset, None, :]
                        Jac[n_cld_ij+n_cld_bdry+cid, 0, i_limit_offset] = - self.e_n_limit_for_Jac[bs_idx, n_o+i_limit_offset, None, :]
                        if i_limit_offset >= 1:
                            Jac[n_cld_ij+n_cld_bdry+cid, 1, i_limit_offset-1] = self.e_t_limit_for_Jac[bs_idx, n_o+i_limit_offset, None, :]
                        Jac[n_cld_ij+n_cld_bdry+cid, 1, i_limit_offset] = - self.e_t_limit_for_Jac[bs_idx, n_o+i_limit_offset, None, :]
                    else: # minimum stretch
                        if i_limit_offset >= 1:
                            Jac[n_cld_ij+n_cld_bdry+cid, 0, i_limit_offset-1] = - self.e_n_limit_for_Jac[bs_idx, n_o+i_limit_offset, None, :]
                        Jac[n_cld_ij+n_cld_bdry+cid, 0, i_limit_offset] = self.e_n_limit_for_Jac[bs_idx, n_o+i_limit_offset, None, :]
                        if i_limit_offset >= 1:
                            Jac[n_cld_ij+n_cld_bdry+cid, 1, i_limit_offset-1] = - self.e_t_limit_for_Jac[bs_idx, n_o+i_limit_offset, None, :]
                        Jac[n_cld_ij+n_cld_bdry+cid, 1, i_limit_offset] = self.e_t_limit_for_Jac[bs_idx, n_o+i_limit_offset, None, :]
                    
            return Jac
        elif d == 3:
            e_n_bdry = torch.tensor([[0, -1, 0]]).type_as(x) # (3)
            e_t1_bdry = torch.tensor([[-1, 0, 0]]).type_as(x)
            e_t2_bdry = torch.tensor([[0, 0, -1]]).type_as(x)
            c_til_bdry_1 = self.contact_c_tilde[bs_idx] # (1, 4)
            Jac = torch.zeros([n_cld, d, n_o, n_p, d], device=x.device, dtype=x.dtype)
            for cid, (i, _) in enumerate(cld_bdry_ids):
                Jac[n_cld_ij+cid, 0, i, :] = - c_til_bdry_1[cid, :, None] * e_n_bdry[cid, None, :]
                Jac[n_cld_ij+cid, 1, i, :] = - c_til_bdry_1[cid, :, None] * e_t1_bdry[cid, None, :]
                Jac[n_cld_ij+cid, 2, i, :] = - c_til_bdry_1[cid, :, None] * e_t2_bdry[cid, None, :]
            return Jac
        else:
            raise NotImplementedError

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
