import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

class ImpulseSolver(object):
    def __init__(
        self, dt, n_o, n_p, d, ls, bdry_lin_coef, check_collision, cld_2did_to_1did, DPhi, delta=None
    ):
        assert delta is not None or n_p == 1
        self.dt = dt
        self.n_o = n_o
        self.n_p = n_p
        self.n = n_o * n_p
        self.d = d
        self.cld_2did_to_1did = cld_2did_to_1did
        self.bdry_lin_coef = bdry_lin_coef
        self.ls = ls
        self.check_collision = check_collision
        self.DPhi = DPhi
        self.delta = delta

    def to(self, device, dtype):
        self.bdry_lin_coef = self.bdry_lin_coef.to(device=device, dtype=dtype)
        self.ls = self.ls.to(device=device, dtype=dtype)
        self.delta = self.delta.to(device=device, dtype=dtype) if self.delta is not None else None
        return self

    def add_impulse(self, xv, mus, cors, Minv):
        bs = xv.shape[0]
        n, d = self.n, self.d
        x = xv.reshape(bs, 2, n, d)[:, 0]
        v = xv.reshape(bs, 2, n, d)[:, 1]
        # x, v = xv.reshape(bs, 2, n, d).unbind(dim=1)
        is_cld, is_cld_ij, is_cld_bdry, dist_ij, dist_bdry = self.check_collision(x)
        if is_cld.sum() == 0:
            return xv, is_cld

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
            
            mu = mus[self.cld_2did_to_1did(cld_ij_ids, cld_bdry_ids)]
            cor = cors[self.cld_2did_to_1did(cld_ij_ids, cld_bdry_ids)]

            # get equality constraints
            DPhi = self.DPhi(x[bs_idx:bs_idx+1], v[bs_idx:bs_idx+1]) # 

            if DPhi.shape[-1] != 0: # exist equality constraints
                new_v = self.update_v_w_J_e(bs_idx, v, Minv, Jac, Jac_v, v_star, mu, cor, DPhi)
            else:                   # no equality constraints
                new_v = self.update_v_wo_J_e(bs_idx, v, Minv, Jac, Jac_v, v_star, mu, cor)

        return torch.stack([x, new_v], dim=1).reshape(bs, -1), is_cld

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

    def update_v_w_J_e(self, bs_idx, v, Minv, Jac, Jac_v, v_star, mu, cor, DPhi):
        n_cld, d, n_o, n_p, _ = Jac.shape
        n = n_o*n_p
        C = DPhi.shape[-1]
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
        if not torch.allclose(e, (e>0.5).to(dtype=e.dtype), atol=1e-6):
            print(f"warning: eigenvalues is {e}")
        L_V_s = (L @ V[:, :n*d-C].reshape(n, d*(n*d-C))).reshape(n*d, n*d-C) # (n*d, n*d-C)
        V_s_T_L_T_Jac_T = L_V_s.t() @ Jac.reshape(n_cld*d, n*d).t() # (n*d-C, n_cld*d)
        A_decom = V_s_T_L_T_Jac_T
        # compression phase impulse
        impulse, impulse_star = self.solve_impulse(A_decom, Jac_v, v_star, mu, n_cld, d, target_impulse=None)
        # velocity after compression phase (before retitution phase)
        M_hat_inv = L_V_s @ L_V_s.t() #(n*d, n*d)
        M_hat_inv_Jac_T = M_hat_inv @ Jac.reshape(n_cld*d, n*d).t() # (n*d, n_cld*d)
        dv_comp = (M_hat_inv_Jac_T @ impulse_star).reshape(n, d)
        v_prev_r = v[bs_idx] + dv_comp
        # restitution phase impulse
        Jac_v_prev_r = Jac.reshape(n_cld*d, n*d) @ v_prev_r.reshape(n*d, 1) # n_cld*d, 1
        target_impulse = (cor.reshape(-1) * impulse.reshape(n_cld, d)[:, 0]).reshape(n_cld, 1)
        impulse_r, impulse_star_r = self.solve_impulse(A_decom, Jac_v_prev_r, v_star, mu, n_cld, d, target_impulse=target_impulse)
        # velocity after restitution phase
        dv_rest = (M_hat_inv_Jac_T @ impulse_r).reshape(n, d)
        dv = torch.zeros_like(v)
        dv[bs_idx] = dv_comp + dv_rest
        return v + dv

    def update_v_wo_J_e(self, bs_idx, v, Minv, Jac, Jac_v, v_star, mu, cor):
        n_cld, d, n_o, n_p, _ = Jac.shape
        n = n_o * n_p
        # calculate A_decom
        Minv_sqrt = torch.cholesky(Minv) # (n, n)
        Minv_sqrt_Jac_T = (Minv_sqrt @ Jac.reshape(n_cld*d, n*d).t().reshape(n, d*n_cld*d)).reshape(n*d, n_cld*d)
        A_decom = Minv_sqrt_Jac_T
        # compression phase impulse
        impulse, impulse_star = self.solve_impulse(A_decom, Jac_v, v_star, mu, n_cld, d, target_impulse=None)
        # velocity after compression phase (before retitution phase)
        Minv_Jac_T = (Minv @ Jac.reshape(n_cld*d, n*d).t().reshape(n, d*n_cld*d)).reshape(n*d, n_cld*d)
        dv_comp = (Minv_Jac_T @ impulse_star).reshape(n, d)
        v_prev_r = v[bs_idx] + dv_comp
        # restitution phase impulse
        Jac_v_prev_r = Jac.reshape(n_cld*d, n*d) @ v_prev_r.reshape(n*d, 1) # n_cld*d, 1
        target_impulse = (cor.reshape(-1) * impulse.reshape(n_cld, d)[:, 0]).reshape(n_cld, 1)
        impulse_r, impulse_star_r = self.solve_impulse(A_decom, Jac_v_prev_r, v_star, mu, n_cld, d, target_impulse=target_impulse)
        # velocity after restitution phase
        dv_rest = (Minv_Jac_T @ impulse_r).reshape(n, d)
        dv = torch.zeros_like(v)
        dv[bs_idx] = dv_comp + dv_rest
        return v + dv

    def solve_impulse(self, A_decom, v_, v_star, mu, n_cld, d, target_impulse):
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
        target_impulse_p = cp.Parameter((n_cld, 1))

        objective = cp.Minimize(0.5 * cp.sum_squares(A_decom_p @ f) + cp.sum(cp.multiply(f, v_p)) - cp.sum(cp.multiply(f, v_star_p)))
        constraints = [cp.SOC(cp.multiply(mu_p[i], f[i*d]), f[i*d+1:i*d+d]) for i in range(n_cld)] + \
                        [f[i*d] >= target_impulse_p[i] for i in range(n_cld)]
        problem = cp.Problem(objective, constraints)
        cvxpylayer = CvxpyLayer(problem, parameters=[A_decom_p, v_p, v_star_p, mu_p, target_impulse_p], variables=[f])
        if target_impulse is None:
            target_impulse = torch.zeros(n_cld, 1, dtype=v_.dtype, device=v_.device)
        impulse, = cvxpylayer(A_decom, v_.reshape(-1, 1), torch.zeros_like(v_star.reshape(-1, 1)), mu.reshape(-1, 1), target_impulse.reshape(-1, 1))
        impulse_star, = cvxpylayer(A_decom, v_.reshape(-1, 1), v_star.reshape(-1, 1), mu.reshape(-1, 1), target_impulse.reshape(-1, 1))
        return impulse, impulse_star