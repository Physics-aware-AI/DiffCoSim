import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation

class Animation():
    def __init__(self, qt, body=None):
        self.qt = qt.detach().cpu().numpy()
        T, n, d = qt.shape
        assert d in (2, 3)
        self.fig = plt.figure()
        self.ax = self.fig.add_axes([0, 0, 1, 1], projection='3d') if d==3 else self.fig.add_axes([0,0,1,1])

        xyzmin = self.qt.min(0).min(0)
        xyzmax = self.qt.max(0).max(0)
        delta = xyzmax - xyzmin
        lower = xyzmin - 0.1 * delta; upper = xyzmax + 0.1 * delta
        self.ax.set_xlim((min(lower), max(upper)))
        self.ax.set_ylim((min(lower), max(upper)))
        if d==3: self.ax.set_zlim((min(lower), max(upper)))
        if d!=3: self.ax.set_aspect("equal")

        empty = d * [[]]
        # self.colors = np.random.choice([f"C{i}" for i in range(15)], size=n, replace=False)
        self.colors = [f"C{i}" for i in range(15)]
        self.objects = {
            'pts': sum([self.ax.plot(*empty, ms=6, color=self.colors[i]) for i in range(n)], []),
            'trails': sum([self.ax.plot(*empty, "-", color=self.colors[i]) for i in range(n)], [])
        }

    def init(self):
        empty = np.array(2 * [[]])
        for obj in self.objects.values():
            for elem in obj:
                elem.set_data(*empty)
                if self.qt.shape[-1]==3: elem.set_3d_properties([])
        return sum(self.objects.values(), [])
    
    def update(self, i=0):
        T, n, d = self.qt.shape
        qt = self.qt.reshape(T, self.n_o, self.n_p, d)
        trail_len = 150
        for j in range(self.n_o):
            # draw trails
            xyz = qt[max(i-trail_len, 0): i+1, j, 0, :]
            self.objects["trails"][j].set_data(*xyz[...,:2].T)
            if d==3: self.objects["trails"][j].set_3d_properties(xyz[...,2].T)
            # draw points
            self.objects['pts'][j].set_data(*xyz[-1:,...,:2].T)
            if d==3: self.objects['pts'][j].set_3d_properties(xyz[-1:,...,2].T)
        return sum(self.objects.values(), [])

    def animate(self):
        return animation.FuncAnimation(self.fig, self.update, frames=self.qt.shape[0],
                    interval=33, init_func=self.init, blit=True,)#.save("test.gif")#.to_html5_video()

def Linear(chin, chout, zero_bias=False, orthogonal_init=False):
    linear = nn.Linear(chin, chout)
    if zero_bias:
        torch.nn.init.zeros_(linear.bias)
    if orthogonal_init:
        torch.nn.init.orthogonal_(linear.weight)
    return linear

def mlp(sizes, activation, output_activation=nn.Identity, orthogonal_init=True):
    layers = []
    for i in range(len(sizes)-1):
        act = activation if i < len(sizes)-2 else output_activation
        layers += [Linear(sizes[i], sizes[i+1], orthogonal_init=orthogonal_init), act()]
    return nn.Sequential(*layers)

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class CosSin(nn.Module):
    def __init__(self, q_ndim, angular_dims, only_q=True):
        super().__init__()
        self.q_ndim = q_ndim
        self.angular_dims = tuple(angular_dims)
        self.non_angular_dims = tuple(set(range(q_ndim)) - set(angular_dims))
        self.only_q = only_q

    def forward(self, q_or_qother):
        if self.only_q:
            q = q_or_qother
        else:
            q, other = q_or_qother.chunk(2, dim=-1)
        assert q.shape[-1] == self.q_ndim
        q_angular = q[..., self.angular_dims]
        q_not_angular = q[..., self.non_angular_dims]
        cos_ang_q, sin_ang_q = q_angular.cos(), q_angular.sin()
        q = torch.cat([cos_ang_q, sin_ang_q, q_not_angular], dim=-1)

        if self.only_q:
            res = q
        else:
            res = torch.cat([q, other], dim=-1)
        return res

def cross_matrix(k):
    """Application of hodge star on R3, mapping Λ^1 R3 -> Λ^2 R3"""
    K = torch.zeros(*k.shape[:-1],3,3,device=k.device,dtype=k.dtype)
    K[...,0,1] = -k[...,2]
    K[...,0,2] = k[...,1]
    K[...,1,0] = k[...,2]
    K[...,1,2] = -k[...,0]
    K[...,2,0] = -k[...,1]
    K[...,2,1] = k[...,0]
    return K

def uncross_matrix(K):
    k = torch.zeros(*K.shape[:-1],device=K.device,dtype=K.dtype)
    k[...,0] = (K[...,2,1] - K[...,1,2])/2
    k[...,1] = (K[...,0,2] - K[...,2,0])/2
    k[...,2] = (K[...,1,0] - K[...,0,1])/2
    return k


def eulerdot_to_omega_matrix(euler):
    """(*bsT, 3) -> (*bsT, 3, 3) matrix"""
    *bsT,_ = euler.shape
    M = torch.zeros(*bsT,3,3,device=euler.device,dtype=euler.dtype)
    phi,theta,psi = euler.unbind(-1)
    M[...,0,0] = theta.sin()*psi.sin()
    M[...,0,1] = psi.cos()
    M[...,1,0] = theta.sin()*psi.cos()
    M[...,1,1] = -psi.sin()
    M[...,2,0] = theta.cos()
    M[...,2,2] = 1
    return M

def euler_to_frame(euler_and_dot):
    """ input: (*bsT, 2, 3)
        output: (*bsT, 2, 3, 3) """
    *bsT, _, _ = euler_and_dot.shape
    euler, eulerdot = euler_and_dot.unbind(dim=-2) # (*bsT, 3)
    omega = (eulerdot_to_omega_matrix(euler) @ eulerdot.unsqueeze(-1)).squeeze(-1) # (*bsT, 3)
    RT_Rdot = cross_matrix(omega) 
    # Rdot_RT = cross_matrix(omega) # (*bsT, 3, 3)
    R = Rotation.from_euler("ZXZ", euler.reshape(-1, 3).detach().cpu().numpy()).as_matrix()
    R = torch.from_numpy(R).reshape(*bsT, 3, 3).to(euler.device, euler.dtype)
    Rdot = R @ RT_Rdot 
    # Rdot = Rdot_RT @ R
    return torch.stack([R, Rdot], dim=-3).transpose(-2, -1) # (bs, 2, d, n) -> (bs, 2, n, d)

def frame_to_euler(frame):
    """ input: (*bsT, 2, 3, 3) output: (*bsT, 2, 3) """
    *bsT, _, _, _ = frame.shape
    R, Rdot = frame.transpose(-2, -1).unbind(-3) # (*bsT, 3, 3)
    omega = uncross_matrix(R.transpose(-2, -1) @ Rdot) 
    # omega = uncross_matrix(Rdot @ R.transpose(-2, -1)) # (*bsT, 3)
    angles = Rotation.from_matrix(R.reshape(-1, 3, 3).detach().cpu().numpy()).as_euler("ZXZ") 
    angles = torch.from_numpy(angles).reshape(*bsT, 3).to(R.device, R.dtype) # (*bsT, 3)
    eulerdot = torch.solve(omega.unsqueeze(-1), eulerdot_to_omega_matrix(angles))[0].squeeze(-1) # (*bsT, 3)
    return torch.stack([angles, eulerdot], dim=-2) # (*bsT, 2, 3)

def com_euler_to_bodyX(com_euler):
    """ input (*bsT, 2, 6), output (*bsT, 2, 4, 3) """
    com = com_euler[..., :3] # (*bsT, 2, 3)
    frame = euler_to_frame(com_euler[..., 3:]) # (*bsT, 2, 3, 3)
    # in C frame, com would be zero
    shifted_frame = frame + com[..., None, :]
    return torch.cat([com[..., None, :], shifted_frame], dim=-2)

def bodyX_to_com_euler(X):
    """ input: (*bsT, 2, 4, 3) output: (*bsT, 2, 6) """
    com = X[..., 0, :] # (*bsT, 2, 3)
    euler = frame_to_euler(X[..., 1:, :] - com[..., None, :]) # (*bsT, 2, 3, 3) -> (*bsT, 2, 3)
    return torch.cat([com, euler], dim=-1)