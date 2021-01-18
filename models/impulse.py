import fsspec, os
import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from diffcp.cone_program import SolverError
from symeig import symeig

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ImpulseSolver(nn.Module):
    def __init__(
        self, dt, n_o, n_p, d, 
        check_collision, cld_2did_to_1did, DPhi,  
        ls, bdry_lin_coef, delta=None, get_limit_e_div_l=None,
        save_dir=os.path.join(PARENT_DIR, "tensors")
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
        self.get_limit_e_div_l = get_limit_e_div_l
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

        if is_cld_limit.shape[1] > 0:
            self.e_n_limit_div_l, self.e_t_limit_div_l = self.get_limit_e_div_l(x)

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
                dv = self.get_dv_w_J_e(bs_idx, v, Minv, Jac, Jac_v, v_star, mu, cor, DPhi)
            else:                   # no equality constraints
                dv = self.get_dv_wo_J_e(bs_idx, v, Minv, Jac, Jac_v, v_star, mu, cor)
            new_v = new_v + dv

        return torch.stack([x, new_v], dim=1).reshape(bs, -1), is_cld

    def get_contact_Jacobian(self, bs_idx, x, cld_ij_ids, cld_bdry_ids, cld_limit_ids):
        bs, n_o, n_p, d = x.shape[0], self.n_o, self.n_p, self.d
        n_cld_ij = len(cld_ij_ids) ; n_cld_bdry = len(cld_bdry_ids)
        n_cld_limit = len(cld_limit_ids)
        n_cld = n_cld_ij + n_cld_bdry + n_cld_limit

        x = x.reshape(bs, n_o, n_p, d)
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
            if t_limit == 0:
                Jac[n_cld_ij+n_cld_bdry+cid, 0, i_limit] += self.e_n_limit_div_l[bs_idx, i_limit, None, :]
                Jac[n_cld_ij+n_cld_bdry+cid, 0, i_limit+1] += - self.e_n_limit_div_l[bs_idx, i_limit+1, None, :]
                Jac[n_cld_ij+n_cld_bdry+cid, 1, i_limit] += self.e_t_limit_div_l[bs_idx, i_limit, None, :]
                Jac[n_cld_ij+n_cld_bdry+cid, 1, i_limit+1] += - self.e_t_limit_div_l[bs_idx, i_limit+1, None, :]
            else:
                Jac[n_cld_ij+n_cld_bdry+cid, 0, i_limit] += - self.e_n_limit_div_l[bs_idx, i_limit, None, :]
                Jac[n_cld_ij+n_cld_bdry+cid, 0, i_limit+1] += self.e_n_limit_div_l[bs_idx, i_limit+1, None, :]
                Jac[n_cld_ij+n_cld_bdry+cid, 1, i_limit] += - self.e_t_limit_div_l[bs_idx, i_limit, None, :]
                Jac[n_cld_ij+n_cld_bdry+cid, 1, i_limit+1] += self.e_t_limit_div_l[bs_idx, i_limit+1, None, :]

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

    def get_dv_w_J_e(self, bs_idx, v, Minv, Jac, Jac_v, v_star, mu, cor, DPhi):
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
        e, V = symeig(Q)
        if not torch.allclose(e, (e>0.5).to(dtype=e.dtype), atol=1e-6):
            print(f"warning: eigenvalues is {e}")
        L_V_s = (L @ V[:, :n*d-C].reshape(n, d*(n*d-C))).reshape(n*d, n*d-C) # (n*d, n*d-C)
        V_s_T_L_T_Jac_T = L_V_s.t() @ Jac.reshape(n_cld*d, n*d).t() # (n*d-C, n_cld*d)
        A_decom = V_s_T_L_T_Jac_T
        # make sure v_star is "valid", avoid unbounded cvx problem
        v_star_c = self.get_v_star_c_w_J_e(n_cld, n, d, v_star, Jac.reshape(n_cld*d, n*d), J_e)

        # Jac_T_v_star = Jac.reshape(n_cld*d, n*d).t() @ v_star.reshape(n_cld*d, 1) # (n*d, 1)
        # J_e_Jac_T_v_star = J_e @ Jac_T_v_star # (C, 1)
        # v_right = J_e_T @ torch.solve(J_e_Jac_T_v_star, J_e @ J_e_T)[0] # (n*d, 1)
        # v_star_c = Jac.reshape(n_cld*d, n*d) @ (Jac_T_v_star - v_right) # (n_cld*d, 1)
        # compression phase impulse
        try:
            impulse = self.solve_compression_impulse(
                A_decom, Jac_v, mu, n_cld, d
            )
        except SolverError as e:
            self.save(e, "comp", bs_idx, v, Minv, Jac, Jac_v, v_star, mu, cor, DPhi)
            impulse = torch.zeros(n_cld*d, 1).type_as(v)
        # velocity after compression phase (before retitution phase)
        M_hat_inv = L_V_s @ L_V_s.t() #(n*d, n*d)
        M_hat_inv_Jac_T = M_hat_inv @ Jac.reshape(n_cld*d, n*d).t() # (n*d, n_cld*d)
        dv_comp = (M_hat_inv_Jac_T @ impulse).reshape(n, d)
        v_prev_r = v[bs_idx] + dv_comp
        # restitution phase impulse
        Jac_v_prev_r = Jac.reshape(n_cld*d, n*d) @ v_prev_r.reshape(n*d, 1) # n_cld*d, 1
        target_impulse = (cor.reshape(-1) * impulse.reshape(n_cld, d)[:, 0]).reshape(n_cld, 1)
        try:
            impulse_star_r = self.solve_restitution_impulse(
                A_decom, Jac_v_prev_r, v_star_c, mu, n_cld, d, target_impulse=target_impulse
            )
        except SolverError as e:
            self.save(e, "rest", bs_idx, v, Minv, Jac, Jac_v, v_star, mu, cor, DPhi)
            impulse_star_r = torch.zeros(n_cld*d, 1).type_as(v)
        # velocity after restitution phase, compensate penetration
        dv_rest = (M_hat_inv_Jac_T @ impulse_star_r).reshape(n, d)
        dv = torch.zeros_like(v)
        dv[bs_idx] = dv_comp + dv_rest
        return dv

    def get_v_star_c_w_J_e(self, n_cld, n, d, v_star, J, J_e):
        if n_cld > n:
            # (J J_T - J J_e_T (J_e J_e_T)^-1 J_e J_T) * v_star
            J_T_v_star = J.t() @ v_star.reshape(n_cld*d, 1)
            J_T_J__inv_J_T_v_star = torch.solve(J_T_v_star, J.t() @ J)[0]
            J_e__J_T_J__inv_J_T_v_star = J_e @ J_T_J__inv_J_T_v_star
            v_right = J_e.t() @ torch.solve(J_e__J_T_J__inv_J_T_v_star, J_e @ J_e.t())[0]
            v_star_c = J @ (J_T_J__inv_J_T_v_star - v_right)
        else:
            J_J_T__inv_v_star = torch.solve(v_star.reshape(n_cld*d, 1), J @ J.t())[0]
            J_T__J_J_T__inv_v_star = J.t() @ J_J_T__inv_v_star
            J_e_J_T__J_J_T__inv_v_star = J_e @ J_T__J_J_T__inv_v_star
            v_right = J_e.t() @ torch.solve(J_e_J_T__J_J_T__inv_v_star, J_e @ J_e.t())[0]
            v_star_c = v_star.reshape(n_cld*d, 1) - J @ v_right
        return v_star_c

    def get_dv_wo_J_e(self, bs_idx, v, Minv, Jac, Jac_v, v_star, mu, cor):
        n_cld, d, n_o, n_p, _ = Jac.shape
        n = n_o * n_p
        # calculate A_decom
        Minv_sqrt = torch.cholesky(Minv) # (n, n)
        Minv_sqrt_Jac_T = (Minv_sqrt @ Jac.reshape(n_cld*d, n*d).t().reshape(n, d*n_cld*d)).reshape(n*d, n_cld*d)
        A_decom = Minv_sqrt_Jac_T
        # make sure v_star is "valid", avoid unbounded cvx problem
        if n_cld > n: # Jac is a tall matrix
            # v_star_c = J (J_T J)^-1 J_T v_star
            v_star_euc = Jac.reshape(n_cld*d, n*d).t() @ v_star.reshape(n_cld*d, 1) # (n*d, 1)
            v_star_euc = torch.solve(v_star_euc, Jac.reshape(n_cld*d, n*d).t() @ Jac.reshape(n_cld*d, n*d))[0]
            v_star_c = Jac.reshape(n_cld*d, n*d) @ v_star_euc
        else:
            v_star_c = v_star
        # compression phase impulse
        try:
            impulse = self.solve_compression_impulse(
                A_decom, Jac_v, mu, n_cld, d
            )
        except SolverError as e:
            self.save(e, "comp", bs_idx, v, Minv, Jac, Jac_v, v_star, mu, cor)
            impulse = torch.zeros(n_cld*d, 1).type_as(v)
        # velocity after compression phase (before retitution phase)
        Minv_Jac_T = (Minv @ Jac.reshape(n_cld*d, n*d).t().reshape(n, d*n_cld*d)).reshape(n*d, n_cld*d)
        dv_comp = (Minv_Jac_T @ impulse).reshape(n, d)
        v_prev_r = v[bs_idx] + dv_comp
        # restitution phase impulse
        Jac_v_prev_r = Jac.reshape(n_cld*d, n*d) @ v_prev_r.reshape(n*d, 1) # n_cld*d, 1
        target_impulse = (cor.reshape(-1) * impulse.reshape(n_cld, d)[:, 0]).reshape(n_cld, 1)
        try:
            impulse_star_r = self.solve_restitution_impulse(
                A_decom, Jac_v_prev_r, v_star_c, mu, n_cld, d, target_impulse=target_impulse
            )
        except SolverError as e:
            self.save(e, "rest", bs_idx, v, Minv, Jac, Jac_v, v_star, mu, cor)
            impulse_star_r = torch.zeros(n_cld*d, 1).type_as(v)
        # velocity after restitution phase
        dv_rest = (Minv_Jac_T @ impulse_star_r).reshape(n, d)
        dv = torch.zeros_like(v)
        dv[bs_idx] = dv_comp + dv_rest
        return dv

    def solve_compression_impulse(self, A_decom, v_, mu, n_cld, d):
        # v_: (n_cld*d, 1)
        n_cld_d = v_.shape[0]
        f = cp.Variable((n_cld_d, 1))
        A_decom_p = cp.Parameter(A_decom.shape) # Todo
        v_p = cp.Parameter((n_cld_d, 1))
        mu_p = cp.Parameter((mu.shape[0], 1)) 

        objective = cp.Minimize(0.5 * cp.sum_squares(A_decom_p @ f) + cp.sum(cp.multiply(f, v_p)))
        constraints = [cp.SOC(cp.multiply(mu_p[i], f[i*d]), f[i*d+1:i*d+d]) for i in range(n_cld)] + \
                        [f[i*d] >= 0 for i in range(n_cld)]
        problem = cp.Problem(objective, constraints)
        cvxpylayer = CvxpyLayer(problem, parameters=[A_decom_p, v_p, mu_p], variables=[f])

        impulse, = cvxpylayer(
            A_decom, 
            v_.reshape(-1, 1), 
            mu.reshape(-1, 1), 
        )
        return impulse

    def solve_restitution_impulse(self, A_decom, v_, v_star, mu, n_cld, d, target_impulse):
        # v_: (n_cld*d, 1)
        n_cld_d = v_.shape[0]
        f = cp.Variable((n_cld_d, 1))
        A_decom_p = cp.Parameter(A_decom.shape) # Todo
        v_p = cp.Parameter((n_cld_d, 1))
        v_star_p = cp.Parameter((n_cld_d, 1))
        mu_p = cp.Parameter((mu.shape[0], 1)) 
        target_impulse_p = cp.Parameter((n_cld, 1))

        objective = cp.Minimize(0.5 * cp.sum_squares(A_decom_p @ f) + cp.sum(cp.multiply(f, v_p)) - cp.sum(cp.multiply(f, v_star_p)))
        # the second line is to avoid negative target_impulse due to numerical error
        constraints = [cp.SOC(cp.multiply(mu_p[i], f[i*d]), f[i*d+1:i*d+d]) for i in range(n_cld)] + \
                        [f[i*d] >= 0 for i in range(n_cld)] + \
                        [f[i*d] >= target_impulse_p[i] for i in range(n_cld)]
        problem = cp.Problem(objective, constraints)
        cvxpylayer = CvxpyLayer(problem, parameters=[A_decom_p, v_p, v_star_p, mu_p, target_impulse_p], variables=[f])
        impulse_star, = cvxpylayer(
            A_decom, 
            v_.reshape(-1, 1), 
            v_star.reshape(-1, 1), 
            mu.reshape(-1, 1), 
            target_impulse.reshape(-1, 1)
        )
        return impulse_star

    def save(self, error, mode, *args):
        version = f"tensors_{self._get_next_version()}"
        dir0 = os.path.join(self.save_dir, version)
        os.makedirs(dir0, exist_ok=True)
        filepath = os.path.join(dir0, f"{mode}.pt")
        torch.save(args, filepath)
        print(error)
        print(f"save tensors at {filepath}")
        
    def _get_next_version(self):
        root_dir = self.save_dir
        fs = fsspec.filesystem("file")

        if not fs.isdir(root_dir):
            return 0
        existing_versions = []
        for listing in fs.listdir(root_dir):
            d = listing["name"]
            bn = os.path.basename(d)
            if fs.isdir(d) and bn.startswith("tensors_"):
                dir_ver = bn.split("_")[1].replace('/', '')
                existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1
