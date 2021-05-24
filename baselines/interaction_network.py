import torch.nn as nn
import torch

from utils import mlp

class InteractionNetwork(nn.Module):
    def __init__(
        self,
        body,
        R_net_hidden_size,
        O_net_hidden_size,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32,
    ):
        super(InteractionNetwork, self).__init__()
        self.body = body
        self.D_S = 2*body.d
        self.D_X = body.d
        self.D_P = body.d

        if "BM" in body.kwargs_file_name:
            self.N_O = body.n_o+4
            self.N_R = body.n_o * (body.n_o-1) + 4 * body.n_o
            self.D_A = 2 # inv_mass and ls
            self.D_R = 2 # mu and cor
            self.has_ls = True
            # 4, 2
            self.walls = torch.tensor([[0.0, 0.5],
                                        [0.5, 0.0],
                                        [1.0, 0.5],
                                        [0.5, 1.0]], device=device, dtype=dtype)
            self.populate_relation_BM()
        elif "BD" in body.kwargs_file_name:
            self.N_O = body.n+4
            self.N_R = body.n_o * (body.n_o-1)  + 4 * body.n_o + 6*body.n_o
            self.D_A = 2 # inv_mass and ls
            self.D_R = 3 # mu, cor, equality_or_contact
            self.has_ls = True
            # 4, 2
            self.walls = torch.tensor([[0.0, 0.5],
                                        [0.5, 0.0],
                                        [1.0, 0.5],
                                        [0.5, 1.0]], device=device, dtype=dtype)
            self.populate_relation_BD()
        elif "CP" in body.kwargs_file_name:
            self.N_O = body.n+2
            self.N_R = body.n_o + 2*(body.n_o-1) + 1
            self.D_A = 2 # inv_mass and ls
            self.D_R = 3 # mu, cor, equality_or_contact
            self.has_ls = True
            # 1, 2
            self.walls = torch.tensor([[0.0, 0.0], [0.0, -1.0]], device=device, dtype=dtype)
            self.populate_relation_CP()
        elif "Gyro" in body.kwargs_file_name:
            self.N_O = body.n+1
            self.N_R = 1 + 6*2
            self.D_A = 2 # inv_mass and ls
            self.D_R = 3 # mu, cor, equality_or_contact
            self.has_ls = True
            # 1, 3
            self.walls = torch.tensor([[0.0, -1.0, 0.0]], device=device, dtype=dtype)
            self.populate_relation_Gyro()
        elif "ER" in body.kwargs_file_name:
            self.N_O = body.n+1
            # angle limit + stretch limit 
            self.N_R = 2*(body.n_o-1)*2 + 2*(body.n_o)*2  
            self.D_A = 2 # inv_mass and ls, we have to put ls here since angle limit does not have have the attribute ls
            self.D_R = 3 # mu, cor, strect_or_limit
            self.has_ls = True
            # 1, 2
            self.walls = torch.tensor([[0.0, 0.0]], device=device, dtype=dtype)
            self.populate_relation_ER()
        elif "Cloth" in body.kwargs_file_name:
            self.N_O = body.n_o + 1
            self.N_R = body.n_c * 4
            self.D_A = 1 # inv_mass 
            self.D_R = 3 # mu, cor, populateed_ls 
            self.has_ls = False
            self.walls = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=dtype)
            self.populate_relation_Cloth()
        else:
            raise NotImplementedError

        self.init_model(self.N_O, self.D_S, self.N_R, self.D_R, self.D_X, self.D_A, self.D_P,
                        R_net_hidden_size, O_net_hidden_size)

    def populate_relation_Cloth(self):
        # R_r: N_R, N_O
        # R_s: N_R, N_O
        # R_a: N_R, D_R
        # ls
        body = self.body
        R_r = torch.zeros(self.N_R, self.N_O)
        R_s = torch.zeros(self.N_R, self.N_O)
        limit_idx_to_o_idx = self.body.limit_idx_to_o_idx
        ptr = 0
        for idx in range(limit_idx_to_o_idx.shape[0]):
            i = limit_idx_to_o_idx[idx, 0]
            j = limit_idx_to_o_idx[idx, 1]
            if idx == 0:
                R_r[ptr, -1] = 1.0 ; R_s[ptr, 0] = 1.0
                R_r[ptr+1, -1] = 1.0 ; R_s[ptr, i] = 1.0
                ptr += 2
                continue
            R_r[ptr, i] = 1.0 ; R_s[ptr, j] = 1.0
            R_r[ptr+1, i] = 1.0 ; R_s[ptr+1, j] = 1.0
            ptr += 2
        assert ptr == 2 * self.body.n_c
        R_r[ptr:] = R_s[:ptr]
        R_s[ptr:] = R_r[:ptr]
        
        self.R_r = R_r
        self.R_s = R_s

        # the orders are mixed up but since they are homogeneous it is fine. 
        mus = torch.cat(
            [body.mus, body.mus, body.mus, body.mus],
            dim=0
        )
        cors = torch.cat(
            [body.cors, body.cors, body.cors, body.cors],
            dim=0
        )
        ls = torch.cat(
            [body.populated_ls, body.populated_ls, body.populated_ls, body.populated_ls],
            dim=0
        )
        self.R_a = torch.stack([mus, cors, ls], dim=1) # N_R, D_R=3
        assert self.R_s.shape == self.R_r.shape == (self.N_R, self.N_O)
        assert self.R_a.shape == (self.N_R, self.D_R)

    def populate_relation_ER(self):
        # R_r: N_R, N_O
        # R_s: N_R, N_O
        # R_a: N_R, D_R
        # ls
        dummy_ids = torch.zeros(0,2, dtype=torch.int)
        body = self.body
        # initiate sub matrices
        R_r = torch.zeros(self.N_R, self.body.n_o+1)
        R_s = torch.zeros(self.N_R, self.body.n_o+1)
        # angle limits
        ptr = 0
        for i in range(body.n_o-1):
            if i == 0:
                R_r[ptr, i] = 1.0 ; R_s[ptr, i+1] = 0.5 ; R_s[ptr, -1] = 0.5
                R_r[ptr+1, i] = 1.0 ; R_s[ptr+1, i+1] = 0.5 ; R_s[ptr+1, -1] = 0.5
                ptr += 2
                continue
            R_r[ptr, i] = 1.0 ; R_s[ptr, i+1] = 0.5 ; R_s[ptr, i-1] = 0.5
            R_r[ptr+1, i] = 1.0 ; R_s[ptr+1, i+1] = 0.5 ; R_s[ptr+1, i-1] = 0.5
            ptr += 2
        assert ptr == 2*(body.n_o -1)
        # stretch limits
        for i in range(body.n_o):
            if i == 0:
                R_r[ptr, -1] = 1.0 ; R_s[ptr, i] = 1.0
                R_r[ptr+1, -1] = 1.0 ; R_s[ptr+1, i] = 1.0
                ptr += 2
                continue
            R_r[ptr, i-1] = 1.0 ; R_s[ptr, i] = 1.0
            R_r[ptr+1, i-1] = 1.0 ; R_s[ptr+1, i] = 1.0
            ptr += 2
        assert ptr == 2*(body.n_o -1) + 2* body.n_o
        R_r[ptr:] = R_s[:ptr]
        R_s[ptr:] = R_r[:ptr]

        self.R_r = R_r
        self.R_s = R_s
        # the orders are mixed up but since they are homogeneous it is fine. 
        mus = torch.cat(
            [body.mus, body.mus, body.mus, body.mus],
            dim=0
        )
        cors = torch.cat(
            [body.cors, body.cors, body.cors, body.cors],
            dim=0
        )
        angle_or_stretch = torch.cat(
            [torch.ones(2*(body.n_o-1)), -torch.ones(2*body.n_o),
            torch.ones(2*(body.n_o-1)), -torch.ones(2*body.n_o)],
            dim=0
        )
        self.R_a = torch.stack([mus, cors, angle_or_stretch], dim=1) # N_R, D_R=3
        self.ls = body.ls
        assert self.R_s.shape == self.R_r.shape == (self.N_R, self.N_O)
        assert self.R_a.shape == (self.N_R, self.D_R)
        assert self.ls.shape[0] == body.n

    def populate_relation_Gyro(self):
        # R_r: N_R, N_O
        # R_s: N_R, N_O
        # R_a: N_R, D_R
        # ls
        body = self.body
        # initiate sub matrices
        Rij_r = torch.zeros(self.N_R, self.body.n_o, self.body.n_p)
        Rij_s = torch.zeros(self.N_R, self.body.n_o, self.body.n_p)
        Rbdry_s = torch.zeros(self.N_R, 1)
        # contact relation
        Rij_r[0, 0, 0] = 1.0
        Rbdry_s[0, 0] = 1.0 
        # equality relation
        ptr = 1
        Rij_r[ptr+0, 0, 0] = 1.0 ; Rij_s[ptr+0, 0, 1] = 1.0
        Rij_r[ptr+1, 0, 1] = 1.0 ; Rij_s[ptr+1, 0, 0] = 1.0
        Rij_r[ptr+2, 0, 0] = 1.0 ; Rij_s[ptr+2, 0, 2] = 1.0
        Rij_r[ptr+3, 0, 2] = 1.0 ; Rij_s[ptr+3, 0, 0] = 1.0
        Rij_r[ptr+4, 0, 1] = 1.0 ; Rij_s[ptr+4, 0, 2] = 1.0
        Rij_r[ptr+5, 0, 2] = 1.0 ; Rij_s[ptr+5, 0, 1] = 1.0
        Rij_r[ptr+6, 0, 0] = 1.0 ; Rij_s[ptr+6, 0, 3] = 1.0
        Rij_r[ptr+7, 0, 3] = 1.0 ; Rij_s[ptr+7, 0, 0] = 1.0
        Rij_r[ptr+8, 0, 1] = 1.0 ; Rij_s[ptr+8, 0, 3] = 1.0
        Rij_r[ptr+9, 0, 3] = 1.0 ; Rij_s[ptr+9, 0, 1] = 1.0
        Rij_r[ptr+10, 0, 2] = 1.0 ; Rij_s[ptr+10, 0, 3] = 1.0
        Rij_r[ptr+11, 0, 3] = 1.0 ; Rij_s[ptr+11, 0, 2] = 1.0

        self.R_r = torch.cat(
            [Rij_r.reshape(self.N_R, self.body.n), torch.zeros(self.N_R, 1)],
            dim=1
        )
        self.R_s = torch.cat(
            [Rij_s.reshape(self.N_R, self.body.n), Rbdry_s],
            dim=1
        )

        mus = torch.cat([body.mus, torch.zeros(12)], dim=0)
        cors = torch.cat([body.cors, torch.zeros(12)], dim=0)
        contact_or_equality = torch.cat(
            [torch.ones(1), -torch.ones(12)], dim=0
        )
        self.R_a = torch.stack([mus, cors, contact_or_equality], dim=1) # N_R, D_R=3
        self.ls = torch.cat(
            [torch.tensor(body.radius).reshape(1), torch.zeros(3)], 
            dim=0
        )
        assert self.R_s.shape == self.R_r.shape == (self.N_R, self.N_O)
        assert self.R_a.shape == (self.N_R, self.D_R)
        assert self.ls.shape[0] == body.n

    def populate_relation_CP(self):
        # R_r: N_R, N_O
        # R_s: N_R, N_O
        # R_a: N_R, D_R
        # ls
        dummy_ids = torch.zeros(0,2, dtype=torch.int)
        body = self.body
        # initiate sub matrices
        Rij_r = torch.zeros(self.N_R, self.body.n_o)
        Rij_s = torch.zeros(self.N_R, self.body.n_o)
        Rbdry_s = torch.zeros(self.N_R, 2)
        ptr = 0
        for i in range(body.n_o):
            id_1d = i
            Rij_r[ptr+id_1d, i] = 1.0
            Rbdry_s[ptr+id_1d, 0] = 1.0 
        ptr = body.n_o
        for i in range(body.n_o-1):
            Rij_r[ptr, i] = 1.0 ; Rij_s[ptr, i+1] = 1.0
            Rij_r[ptr+1, i+1] = 1.0 ; Rij_s[ptr+1, i] = 1.0
            ptr += 2
        # first link
        Rij_r[ptr, 0] = 1.0 ; Rbdry_s[ptr, 1] = 1.0
        assert ptr + 1 == self.N_R

        self.R_r = torch.cat(
            [Rij_r, torch.zeros(self.N_R, 2)],
            dim=1
        )
        self.R_s = torch.cat(
            [Rij_s, Rbdry_s],
            dim=1
        )

        mus = torch.cat(
            [body.mus, torch.zeros(2*body.n_o-1)],
            dim=0
        )
        cors = torch.cat(
            [body.cors, torch.zeros(2*body.n_o-1)],
            dim=0
        )
        contact_or_equality = torch.cat(
            [torch.ones(body.n_o), -torch.ones(2*body.n_o-1)],
            dim=0
        )
        self.R_a = torch.stack([mus, cors, contact_or_equality], dim=1) # N_R, D_R=3
        self.ls = body.ls
        assert self.R_s.shape == self.R_r.shape == (self.N_R, self.N_O)
        assert self.R_a.shape == (self.N_R, self.D_R)
        assert self.ls.shape[0] == body.n

    def populate_relation_BD(self):
        # R_r: N_R, N_O
        # R_s: N_R, N_O
        # R_a: N_R, D_R
        # ls
        dummy_ids = torch.zeros(0,2, dtype=torch.int)
        body = self.body
        n_ij = body.n_o * (body.n_o-1) // 2
        ptr = n_ij
        # initiate sub matrices
        Rij_r = torch.zeros(self.N_R, self.body.n_o, self.body.n_p)
        Rij_s = torch.zeros(self.N_R, self.body.n_o, self.body.n_p)
        Rbdry_s = torch.zeros(self.N_R, 4)
        for i in range(body.n_o):
            for j in range(i+1, body.n_o):
                id_ids = torch.tensor([[i, j]], dtype=torch.int)
                id_1d = self.body.cld_2did_to_1did(id_ids, dummy_ids, dummy_ids)[0]
                Rij_r[id_1d, i, 0] = 1.0
                Rij_s[id_1d, j, 0] = 1.0
                Rij_r[ptr+id_1d, j, 0] = 1.0
                Rij_s[ptr+id_1d, i, 0] = 1.0
        ptr = 2*n_ij
        for i in range(body.n_o):
            for j in range(4):
                id_1d = i * 4 + j
                Rij_r[ptr+id_1d, i, 0] = 1.0
                Rbdry_s[ptr+id_1d, j] = 1.0 
        ptr = 2*n_ij + body.n_o * 4
        for i in range(body.n_o):
            Rij_r[ptr+0, i, 0] = 1.0 ; Rij_s[ptr+0, i, 1] = 1.0
            Rij_r[ptr+1, i, 1] = 1.0 ; Rij_s[ptr+1, i, 0] = 1.0
            Rij_r[ptr+2, i, 0] = 1.0 ; Rij_s[ptr+2, i, 2] = 1.0
            Rij_r[ptr+3, i, 2] = 1.0 ; Rij_s[ptr+3, i, 0] = 1.0
            Rij_r[ptr+4, i, 1] = 1.0 ; Rij_s[ptr+4, i, 2] = 1.0
            Rij_r[ptr+5, i, 2] = 1.0 ; Rij_s[ptr+5, i, 1] = 1.0
            ptr += 6

        self.R_r = torch.cat(
            [Rij_r.reshape(self.N_R, self.body.n), torch.zeros(self.N_R, 4)],
            dim=1
        )
        self.R_s = torch.cat(
            [Rij_s.reshape(self.N_R, self.body.n), Rbdry_s],
            dim=1
        )

        mus = torch.cat(
            [body.mus[0:n_ij], body.mus[0:n_ij], body.mus[n_ij:], torch.zeros(6*body.n_o)],
            dim=0
        )
        cors = torch.cat(
            [body.cors[0:n_ij], body.cors[0:n_ij], body.cors[n_ij:], torch.zeros(6*body.n_o)],
            dim=0
        )
        contact_or_equality = torch.cat(
            [torch.ones(2*n_ij + body.n_o * 4), -torch.ones(body.n_o * 6)],
            dim=0
        )
        self.R_a = torch.stack([mus, cors, contact_or_equality], dim=1) # N_R, D_R=3
        self.ls = torch.cat(
            [body.ls[:, None], torch.zeros(body.n_o, 2).type_as(body.ls)], 
            dim=1
        ).reshape(body.n)
        assert self.R_s.shape == self.R_r.shape == (self.N_R, self.N_O)
        assert self.R_a.shape == (self.N_R, self.D_R)
        assert self.ls.shape[0] == body.n

    def populate_relation_BM(self):
        # R_r: N_R, N_O
        # R_s: N_R, N_O
        # R_a: N_R, D_R
        # ls
        R_r = torch.zeros(self.N_R, self.N_O)
        R_s = torch.zeros(self.N_R, self.N_O)
        dummy_ids = torch.zeros(0,2, dtype=torch.int)
        body = self.body
        n_ij = body.n_o * (body.n_o-1) // 2
        for i in range(body.n_o):
            for j in range(i+1, body.n_o):
                id_ids = torch.tensor([[i, j]], dtype=torch.int)
                id_1d = self.body.cld_2did_to_1did(id_ids, dummy_ids, dummy_ids)[0]
                R_r[id_1d, i] = 1.0
                R_s[id_1d, j] = 1.0
                R_r[n_ij+id_1d, j] = 1.0
                R_s[n_ij+id_1d, i] = 1.0

        n_bdry = body.n_o * 4
        for i in range(body.n_o):
            for j in range(4):
                id_1d = i * 4 + j
                R_r[2*n_ij+id_1d, i] = 1.0
                R_s[2*n_ij+id_1d, body.n_o+j] = 1.0 
        self.R_r = R_r
        self.R_s = R_s

        mus = torch.cat(
            [self.body.mus[0:n_ij], self.body.mus[0:n_ij], self.body.mus[n_ij:]],
            dim=0
        )
        cors = torch.cat(
            [self.body.cors[0:n_ij], self.body.cors[0:n_ij], self.body.cors[n_ij:]],
            dim=0
        )
        self.R_a = torch.stack([mus, cors], dim=1) # N_R, D_R=2
        self.ls = self.body.ls
        assert self.R_s.shape == self.R_r.shape == (self.N_R, self.N_O)
        assert self.R_a.shape == (self.N_R, self.D_R)
        assert self.ls.shape[0] == body.n


    def init_model(self, N_O, D_S, N_R, D_R, D_X, D_A, D_P, D_E=50,
                    R_net_hidden_size=150, O_net_hidden_size=100):
        sizes = [D_S+2*D_A+D_R] + 4 * [R_net_hidden_size] + [D_E]
        self.relational_model = mlp(sizes, activation=nn.ReLU, orthogonal_init=True)
        sizes = [D_S+D_X+D_E] + [O_net_hidden_size] + [D_P]
        self.object_model = mlp(sizes, activation=nn.ReLU, orthogonal_init=True)

    def forward(self, z, inv_moments, f_external):
        """
        input:
        z:, bs, 2, n, d
        inv_moments: (n,)
        f_external: (bs, n, d)

        """
        # add walls
        bs, _, n, d = z.shape
        n_w, d_w = self.walls.shape
        assert n+n_w == self.N_O and d == d_w
        walls = torch.stack([self.walls, torch.zeros_like(self.walls)], dim=0)[None].expand(bs, -1, -1, -1)
        # construct O
        O = torch.cat([z, walls.type_as(z)], dim=2).permute(0, 2, 1, 3).reshape(bs, self.N_O, 2*d)
        # construct X (bs, N_O, D_X=d)
        X = torch.cat([f_external.type_as(z), torch.zeros(bs, n_w, d).type_as(z)], dim=1)
        # construct object attribute matrix N_O, D_A
        padded_Minv = torch.cat([inv_moments.type_as(z), torch.zeros(n_w).type_as(z)], dim=0)[:, None] # N_O, 1
        if self.has_ls:
            padded_ls = torch.cat([self.ls.type_as(z), torch.zeros(n_w).type_as(z)], dim=0)[:, None] # N_O, 1
            A = torch.cat([padded_Minv, padded_ls], dim=1) # N_O, 2
        else:
            A = padded_Minv
        # convert relations type
        R_s = self.R_s.type_as(z)
        R_r = self.R_r.type_as(z)
        R_a = self.R_a.type_as(z)
        R_s_O = R_s @ O # bs, N_R, 2*d
        R_r_O = R_r @ O # bs, N_R, 2*d
        s_a = (R_s @ A)[None].expand(bs, -1, -1) # bs, N_R, D_A
        r_a = (R_r @ A)[None].expand(bs, -1, -1) # bs, N_R, D_A
        R_a = R_a[None].expand(bs, -1, -1)
        
        B = torch.cat([R_r_O-R_s_O, r_a, s_a, R_a], dim=-1) # bs, N_R, D_S+2*D_A+D_R
        E = self.relational_model(B) # bs, N_R, D_E
        C = torch.cat([O, X, R_r.t() @ E], -1) # bs, N_O, D_S+D_X+D_E
        P = self.object_model(C) # bs, N_O, D_P=d
        return P[:, 0:self.body.n]

