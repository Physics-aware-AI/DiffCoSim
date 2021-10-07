"""
code modified from constrained-hamiltonian-neural-networks
https://github.com/mfinzi/constrained-hamiltonian-neural-networks
"""

from systems.rigid_body import RigidBody, BodyGraph
from utils import Animation, com_euler_to_bodyX, bodyX_to_com_euler
from models.impulse import ImpulseSolver
from pytorch_lightning import seed_everything
import numpy as np
import torch
import networkx as nx
from models.impulse_mujoco import ImpulseSolverMujoco
from baselines.lcp.impulse_lcp import ImpulseSolverLCP

class GyroscopeWithWall(RigidBody):
    dt = 0.02
    integration_time = 2
    n = 4 ; d = 3 ; D = 3
    angular_dims = range(3)

    def __init__(
        self, 
        kwargs_file_name="default",
        m=0.1, 
        com=[0,0,0.5], 
        moments=[0.24, 0.24, 0.04],
        mus=[0.0],
        cors=[1.0],
        bdry_lin_coef=[[0, 1, 0, 0.33]],
        is_homo=True,
        offset=0.0,
        radius=0.3,
        is_mujoco_like=False,
        is_lcp_model=False,
        is_lcp_data=False,
        dtype=torch.float64
    ):
        assert not (is_mujoco_like and is_lcp_model)
        self.body_graph = BodyGraph()
        self.kwargs_file_name = kwargs_file_name
        self.m = m
        self.moments = torch.tensor(moments, dtype=torch.float64)
        self.com = torch.tensor(com, dtype=torch.float64)
        self.body_graph.add_extended_body(0, m=m, moments=self.moments, d=3)
        self.body_graph.add_joint(0, -self.com, pos2=torch.tensor([0.,0.,0], dtype=torch.float64))
        self.n_o, self.n_p, self.d = 1, 4, 3
        self.n = self.n_o * self.n_p
        self.bdry_lin_coef = torch.tensor(bdry_lin_coef, dtype=torch.float64)
        self.n_c = 1
        assert len(mus) == len(cors) == 1
        self.mus = torch.tensor(mus, dtype=torch.float64)
        self.cors = torch.tensor(cors, dtype=torch.float64)
        self.is_homo = is_homo
        self.is_mujoco_like = is_mujoco_like
        self.is_lcp_model = is_lcp_model
        self.is_lcp_data = is_lcp_data

        self.delta = torch.tensor([[-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]], dtype=torch.float64) # 3, 4
        self.offset = offset
        self.radius = radius

        if is_lcp_model:
            self.impulse_solver = ImpulseSolverLCP(
                dt = self.dt,
                n_o = self.n_o,
                n_p = self.n_p,
                d = self.d,
                ls = None,
                bdry_lin_coef = self.bdry_lin_coef,
                check_collision = self.check_collision,
                cld_2did_to_1did = self.cld_2did_to_1did,
                DPhi = self.DPhi,
                delta=self.delta,
                get_3d_contact_point_c_tilde=self.get_3d_contact_point_c_tilde
            )
        elif is_mujoco_like:
            self.impulse_solver = ImpulseSolverMujoco(
                dt = self.dt,
                n_o = self.n_o,
                n_p = self.n_p,
                d = self.d,
                ls = None,
                bdry_lin_coef = self.bdry_lin_coef,
                check_collision = self.check_collision,
                cld_2did_to_1did = self.cld_2did_to_1did,
                DPhi = self.DPhi,
                delta=self.delta,
                get_3d_contact_point_c_tilde=self.get_3d_contact_point_c_tilde
            )
        else:
            self.impulse_solver = ImpulseSolver(
                dt = self.dt,
                n_o = self.n_o,
                n_p = self.n_p,
                d = self.d,
                ls = None,
                bdry_lin_coef = self.bdry_lin_coef,
                check_collision = self.check_collision,
                cld_2did_to_1did = self.cld_2did_to_1did,
                DPhi = self.DPhi,
                delta=self.delta,
                get_3d_contact_point_c_tilde=self.get_3d_contact_point_c_tilde
            )

    def __str__(self):
        if self.is_mujoco_like:
            return f"{self.__class__.__name__}{self.kwargs_file_name}_mujoco"
        elif self.is_lcp_data:
            return f"{self.__class__.__name__}{self.kwargs_file_name}_lcp"
        else:
            return f"{self.__class__.__name__}{self.kwargs_file_name}"
        # return f"{self.__class__.__name__}{self.kwargs_file_name}"

    def potential(self, r):
        M = self.M.to(dtype=r.dtype)
        return 9.81 * (M @ r)[..., 2].sum(1)

    def sample_initial_conditions(self, N):
        xv_list = []
        ptr = 0
        while ptr < N:
            eulers = (torch.rand(N, 2, 3, dtype=torch.float64) - 0.5) * 3
            # eulers[:, 0, 1] *= 0.2
            eulers[:, 1, 0] *= 3
            eulers[:, 1, 1] *= 0.2
            eulers[:, 1, 2] = (torch.randint(2, size=(N,), dtype=torch.float64) * 2 -1) * (torch.randn(N) + 7) * 1.5
            xv = self.angle_to_global_cartesian(eulers)
            x = xv[:, 0] # (N, 4, 3)
            is_cld, *_ = self.check_collision(x)
            xv_list.append(xv[torch.logical_not(is_cld)])
            ptr += sum(torch.logical_not(is_cld))
        xv = torch.cat(xv_list, dim=0)[0:N]
        return xv

    def check_collision(self, x):
        bs = x.shape[0]
        is_cld_ij = torch.zeros(bs, 1, 1, dtype=torch.bool, device=x.device)
        dist_ij = torch.zeros(bs, 1, 1).type_as(x)        
        is_cld_limit = torch.zeros(bs, 0, 2, dtype=torch.bool, device=x.device)
        dist_limit = torch.zeros(bs, 0, 2).type_as(x)
        is_cld, is_cld_bdry, dist_bdry = self.check_boundary_collision(x)
        return is_cld, is_cld_ij, is_cld_bdry, is_cld_limit, dist_ij, dist_bdry, dist_limit

    def check_boundary_collision(self, x):
        is_cld_ndry, is_cld_bdry, min_dist, _ = self.get_bdry_cld_and_c_tilde(x)
        return is_cld_ndry, is_cld_bdry, min_dist

    def get_3d_contact_point_c_tilde(self, x):
        _, _, _, c_tilde = self.get_bdry_cld_and_c_tilde(x)
        return c_tilde

    def get_bdry_cld_and_c_tilde(self, x):
        # assume the coefficient are normalized
        bdry_lin_coef = self.bdry_lin_coef.type_as(x)
        delta = self.delta.type_as(x)
        # get the position of the mesh points on the rim of the cone
        assert x.ndim == 3
        bs = x.shape[0]
        x = x.reshape(bs, 1, 4, 3)
        com = x[:, :, 0] # (bs, 1, 3)
        e = delta @ x # (bs, 1, 3, 3)
        angles = torch.linspace(0, 6.28, 21)
        mesh_points = torch.stack(
            [e[:,:,0]*self.radius*a.cos() + e[:,:,1]*self.radius*a.sin() + com+e[:,:,2]*self.offset for a in angles][:-1],
            dim=1,
        ) # (bs, 20, 1, 3)
        mesh_points_one = torch.cat(
            [mesh_points, torch.ones(*mesh_points.shape[:-1], 1).type_as(mesh_points)],
            dim=-1,
        ).unsqueeze(-2) # (bs, 20, 1, 1, 4)
        dist = (mesh_points_one * bdry_lin_coef).sum(-1) # (bs, 20, 1, 1) # n_o, n_bdry
        min_dist, min_dist_idx = torch.min(dist, dim=1) # (bs, 1, 1), (bs, 1, 1)
        is_cld_bdry = min_dist < 0
        cld_angles = angles[min_dist_idx.squeeze()] # (bs, )
        c = torch.stack(
            [self.radius*cld_angles.cos(), self.radius*cld_angles.sin(), self.offset*torch.ones_like(cld_angles)],
            dim=-1
        ) # (bs, 3)
        c_tilde = torch.cat(
            [1 - c.sum(-1, keepdim=True), c], dim=-1
        )
        
        return is_cld_bdry[:,0,0], is_cld_bdry, min_dist, c_tilde.unsqueeze(1)

    def cld_2did_to_1did(self, cld_ij_ids, cld_bdry_ids, cld_limit_ids):
        return [0]

    def angle_to_global_cartesian(self, eulers):
        """ input: (*bsT, 2, 3), output: (*bsT, 2, 4, 3) """
        local_coms = torch.zeros_like(eulers)
        local_com_eulers = torch.cat([local_coms, eulers], dim=-1) # (*bsT, 2, 6)
        bodyX = com_euler_to_bodyX(local_com_eulers) # (*bsT, 2, 4, 3)
        # checked for com is zero, need to understand when com is not zero
        body_attachment = self.body_graph.nodes[0]['joint'][0].to(eulers.device,eulers.dtype) # (3,)
        ct = torch.cat([1-body_attachment.sum()[None],body_attachment]) # (4,)
        global_coords_attachment_point = (bodyX*ct[:,None]).sum(-2,keepdims=True) # (*bsT, 2, 1, 3)
        return bodyX-global_coords_attachment_point

    def global_cartesian_to_angle(self, global_cartesian):
        """ input: (*bsT, 2, 4, 3), output: (*bsT, 2, 3) """
        eulers = bodyX_to_com_euler(global_cartesian)[..., 3:] # (*bsT, 2, 3)
        if eulers.ndim == 4:
            eulers[..., 0, :] = torch.from_numpy(np.unwrap(eulers[..., 0, :].detach().cpu().numpy(), axis=1)).to(eulers.device, eulers.dtype)
        elif eulers.ndim == 3:
            eulers[..., 0, :] = torch.from_numpy(np.unwrap(eulers[..., 0, :].detach().cpu().numpy(), axis=0)).to(eulers.device, eulers.dtype)
        else:
            raise NotImplementedError
        return eulers

    @property
    def animator(self):
        return GyroscopeAnimation


class GyroscopeAnimation(Animation):
    def __init__(self, qt, body):
        super().__init__(qt, body)
        self.body = body
        self.n_o = body.n_o
        self.n_p = body.n_p

        # plot the wall
        corner = 0.6
        xx = np.array([[-corner, corner],
                        [-corner, corner]])
        y_coor = body.bdry_lin_coef[0, 3].numpy()
        yy = np.array([[-y_coor, -y_coor],
                        [-y_coor, -y_coor]])
        zz = np.array([[-corner, -corner],
                        [corner, corner]])
        self.ax.plot_surface(xx, yy, zz, color="lightgray", zorder=-1)
        # plot the cone
        self.cone = [self.ax.plot_surface(xx, yy, zz, zorder=1)]


        # plot the lines
        # self.objects['lines'] = sum([self.ax.plot([],[],[],"-",c="k") for _ in range(4)], [])
        self.objects["lines"] = sum([self.ax.plot([], [], [], "-", c="k", zorder=5) for _ in range(2)], [])
        self.objects['trails'] = sum([self.ax.plot([], [], [], "-", color="teal", zorder=10) for i in range(1)], [])

        # self.ax.view_init(elev=20., azim=10)
        self.ax.view_init(elev=20., azim=-1)
        self.ax.dist = 5.8 
        self.ax.set_xlim3d(-1.2*corner, 1.2*corner)
        self.ax.set_ylim3d(-1.2*corner, 1.2*corner)
        self.ax.set_zlim3d(-1.2*corner, 1.2*corner)
        self.ax.axis("off"),
        self.fig.set_size_inches(11.5, 11.5)

    def update(self, i=0):
        e = self.body.delta.numpy() @ self.qt[i] # (3, 3)
        height = self.body.offset+self.body.com[-1].numpy()
        tip = 0.15
        p0 = e[2] * height
        p1 = p0 + e[0] * self.body.radius
        p2 = p0 + e[2] * tip
        self.objects['lines'][0].set_data(np.array([p0[0], p1[0]]), np.array([p0[1],p1[1]]))
        self.objects['lines'][0].set_3d_properties(np.array([p0[2],p1[2]]))
        self.objects['lines'][1].set_data(np.array([p0[0], p2[0]]), np.array([p0[1],p2[1]]))
        self.objects['lines'][1].set_3d_properties(np.array([p0[2],p2[2]]))

        
        # x, y, z = self.qt[i,0].T
        # self.objects['lines'][0].set_data(np.array([0,x]), np.array([0,y]))
        # self.objects['lines'][0].set_3d_properties(np.array([0,z]))
        # for j in range(1,4):
        #     self.objects['lines'][j].set_data(*self.qt[i, (0, j)].T[:2])
        #     self.objects['lines'][j].set_3d_properties(self.qt[i, (0, j)].T[2])

        # plot cone
        self.cone[0].remove()
        step = 40
        t = np.linspace(0, height, step)
        theta = np.linspace(0, 2*np.pi, step)
        t, theta = np.meshgrid(t, theta)
        R = np.linspace(0.0001, self.body.radius, step)
        xx, yy, zz = [e[0,j]*R*np.cos(theta) + e[1,j]*R*np.sin(theta) + e[2,j]*t for j in [0, 1, 2]]
        self.cone[0] = self.ax.plot_surface(xx, yy, zz, color="orangered", zorder=4)
        # plot trails
        trail_len = 150
        T, n, d = self.qt.shape
        qt = self.qt.reshape(T, self.n_o, self.n_p, d)
        e0 = qt[max(i-trail_len, 0): i+1, 0, 0, :] # trail_len, 3
        e3 = qt[max(i-trail_len, 0): i+1, 0, 3, :] # trail_len, 3
        xyz = e0 + (e3 - e0) * (self.body.offset + tip) 
        self.objects["trails"][0].set_data(*xyz[...,:2].T)
        self.objects["trails"][0].set_3d_properties(xyz[...,2].T)        
        return sum(self.objects.values(),[])