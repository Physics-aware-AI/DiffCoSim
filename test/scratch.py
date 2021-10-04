# import torch

# a = torch.tensor([[1,2],
#                 [3,4],
#                 [5,6],
#                 [7,8],
#                 [9,10],
#                 [11,12]])

# bool_tensor = torch.tensor([[0, 1, 1, 1, 0],
#                             [0, 0, 1, 0, 0],
#                             [0, 0, 0, 0, 1],
#                             [0, 0, 0, 0, 0],
#                             [0, 0, 0, 0, 0]], dtype=torch.bool)

# ids = torch.nonzero(bool_tensor)

# b = a[ids]
# print(a[ids])

# import sympy as sym
# from sympy import *

# x, y = symbols('x y')
# px, py = symbols('px, py')
# D_Phi = Matrix([[x, y, 0, 0], [px, py, x, y]])
# J = Matrix([[0, 0, 1, 0],
#             [0, 0, 0, 1],
#             [-1, 0, 0, 0],
#             [0, -1, 0, 0]])

# I = Matrix([[1, 0, 0, 0],
#             [0, 1, 0, 0],
#             [0, 0, 1, 0],
#             [0, 0, 0, 1]])
# print(D_Phi)
# print(I - J * D_Phi.T * (D_Phi * J * D_Phi.T)**(-1) * D_Phi)

# import cvxpy as cp
# import torch
# from cvxpylayers.torch import CvxpyLayer
# x = cp.Variable((2, 1))
# A_d_p = cp.Parameter((2, 2))
# objective = cp.Minimize(cp.sum_squares(A_d_p @ x))
# problem = cp.Problem(objective)
# layer = CvxpyLayer(problem, parameters=[A_d_p], variables=[x])
# A_d = torch.tensor([[0, 0], [0, 1]])
# solution, = layer(A_d)
# print(solution)
# print(solution.grad)

import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer
x = cp.Variable((2, 1))
A_d_p = cp.Parameter((2, 1))
objective = cp.Minimize(cp.sum_squares(cp.multiply(A_d_p, x)))
problem = cp.Problem(objective)
layer = CvxpyLayer(problem, parameters=[A_d_p], variables=[x])
A_d = torch.tensor([[2], [1]])
solution, = layer(A_d)
print(solution)


######################################
# example of a ball without rotation

# e_n = torch.tensor([0, -1], dtype=torch.float32)
# e_t = torch.tensor([1, 0], dtype=torch.float32)
# Jac = torch.zeros([1, 2, 1, 2], dtype=torch.float32)

# Jac[0, 0, 0, :] = - e_n
# Jac[0, 1, 0, :] = - e_t

# Minv = torch.tensor([1], dtype=torch.float32)
# Minv_Jac_T = (Minv @ Jac.reshape(2, 2).t().reshape(1, 4)).reshape(2, 2)
# A = Jac.reshape(2, 2) @ Minv_Jac_T

##################################3
# no friction
# f = cp.Variable(2)
# A_sqrt = cp.Parameter((2,2))
# v_ = cp.Parameter(2)
# objective = cp.Minimize(0.5 * cp.sum_squares(A_sqrt @ f) + cp.sum(cp.multiply(f, v_)))
# constraints = [f[0] >= 0, f[1] == 0]

# problem = cp.Problem(objective, constraints)
# print(problem.is_dpp())

# cvxpylayer = CvxpyLayer(problem, parameters=[A_sqrt, v_], variables=[f])

# A_sqrt_tch = torch.cholesky(A)
# v_tch = torch.tensor([-2, -1], dtype=torch.float32)

# solution, = cvxpylayer(A_sqrt_tch, v_tch)

# print(solution)

##########################3
# with friction
# f = cp.Variable(2)
# A_sqrt = cp.Parameter((2, 2))
# v_ = cp.Parameter(2)
# mu = cp.Parameter(1)
# objective = cp.Minimize(0.5 * cp.sum_squares(A_sqrt @ f) + cp.sum(cp.multiply(f, v_)))
# constraints = [f >= 0, cp.SOC(mu * f[0], f[1:])]

# problem = cp.Problem(objective, constraints)

# cvxpylayer = CvxpyLayer(problem, parameters=[A_sqrt, v_, mu], variables=[f])

# A_sqrt_tch = torch.cholesky(A)
# v_tch = torch.tensor([-1, -3], dtype=torch.float32)
# mu_tch = torch.tensor([1], dtype=torch.float32)

# solution, = cvxpylayer(A_sqrt_tch, v_tch, mu_tch)

# print(solution)

#######################################3
# problem with two contact at the same time

# e_n = torch.tensor([[0, -1], [1, 0]], dtype=torch.float32)
# e_t = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
# Jac = torch.zeros([2, 2, 1, 2], dtype=torch.float32)

# Jac[:, 0, 0, :] = - e_n
# Jac[:, 1, 0, :] = - e_t

# Minv = torch.tensor([[1]], dtype=torch.float32)
# # Minv_Jac_T = (torch.cholesky(Minv) @ Jac.reshape(4, 2).t().reshape(1, 8)).reshape(2, 4)
# # A = Jac.reshape(4, 2) @ Minv_Jac_T # 4, 4


# Minv_sqrt_Jac_T = (torch.cholesky(Minv) @ Jac.reshape(4, 2).t().reshape(1, 8)).reshape(2, 4)

# A = Minv_sqrt_Jac_T.t() @ Minv_sqrt_Jac_T

##############################
# no friction
# f = cp.Variable((4))
# A_decom = cp.Parameter((2, 4))
# v_ = cp.Parameter((4))

# objective = cp.Minimize(0.5 * cp.sum_squares(A_decom @ f) + cp.sum(cp.multiply(f, v_)))
# constraints = [f[0] >= 0, f[1] == 0, f[2] >= 0, f[3] == 0]

# problem = cp.Problem(objective, constraints)
# print(problem.is_dpp())

# cvxpylayer = CvxpyLayer(problem, parameters=[A_decom, v_], variables=[f])


# v_tch = torch.tensor([-2, -1, -1, 2], dtype=torch.float32)

# solution, = cvxpylayer(Minv_sqrt_Jac_T, v_tch)

# print(solution)


##########################3
# with friction
# def get_impulse(A_decom, v_, v_star, mu, n_cld, d):
#     # v_: (n_cld*d, 1)
#     n_cld_d = v_.shape[0]
#     f = cp.Variable((n_cld_d, 1))
#     A_decom_p = cp.Parameter(A_decom.shape) # Todo
#     v_p = cp.Parameter((n_cld_d, 1))
#     v_star_p = cp.Parameter((n_cld_d, 1))
#     mu_p = cp.Parameter((mu.shape[0], 1)) 

#     objective = cp.Minimize(0.5 * cp.sum_squares(A_decom_p @ f) + cp.sum(cp.multiply(f, v_p)) - cp.sum(cp.multiply(f, v_star_p)))
#     constraints = [f[i*d] >= 0 for i in range(n_cld)] + \
#                     [cp.SOC(cp.multiply(mu_p[i], f[i*d]), f[i*d+1:i*d+d]) for i in range(n_cld)]

#     problem = cp.Problem(objective, constraints)
#     cvxpylayer = CvxpyLayer(problem, parameters=[A_decom_p, v_p, v_star_p, mu_p], variables=[f])

#     impulse, = cvxpylayer(A_decom, v_.reshape(-1, 1), torch.zeros_like(v_star.reshape(-1, 1)), mu.reshape(-1, 1))
#     impulse_star, = cvxpylayer(A_decom, v_.reshape(-1, 1), v_star.reshape(-1, 1), mu.reshape(-1, 1))
#     return impulse, impulse_star


# f = cp.Variable((4, 1))
# A_decom = cp.Parameter((2, 4))
# v_ = cp.Parameter((4, 1))
# mu = cp.Parameter((2, 1))
# objective = cp.Minimize(0.5 * cp.sum_squares(A_decom @ f) + cp.sum(cp.multiply(f, v_)))
# constraints = [f[0] >= 0, f[2] >= 0, cp.SOC(cp.multiply(mu[0], f[0]), f[1:2]), cp.SOC(cp.multiply(mu[1], f[2]), f[3:4])]

# problem = cp.Problem(objective, constraints)

# cvxpylayer = CvxpyLayer(problem, parameters=[A_decom, v_, mu], variables=[f])

# A_sqrt_tch = torch.cholesky(A)
# v_tch = torch.tensor([1, -2], dtype=torch.float32).reshape(2, 1)
# mu_tch = torch.tensor([1, 1], dtype=torch.float32).reshape(2, 1)

# solution, = cvxpylayer(Minv_sqrt_Jac_T, (Jac.reshape(4, 2) @ v_tch).reshape(4, 1), mu_tch)

# print(solution)

# impulse, impulse_star = get_impulse(Minv_sqrt_Jac_T, 
#                                     (Jac.reshape(4, 2) @ v_tch).reshape(4, 1),
#                                     torch.zeros_like((Jac.reshape(4, 2) @ v_tch).reshape(4, 1)),
#                                     mu_tch, 2, 2)
# Minv_Jac_T = (Minv @ Jac.reshape(4, 2).t().reshape(1, 8)).reshape(2, 4)
# v_plus = v_tch + (Minv_Jac_T @ impulse).reshape(2, 1) # special

# impulse_dbl = torch.zeros_like(impulse)
# impulse_dbl[0, 0] = impulse[0, 0] 
# impulse_dbl[2, 0] = impulse[2, 0]

# v_plus_dbl = v_plus + (Minv_Jac_T @ impulse_dbl).reshape(2, 1) # special
# print(f"impulse:{impulse}")
# print(f"impulse_dbl:{impulse_dbl}")
# # print(impulse_star)
# print(f"v_plus: {v_plus}")
# print(f"v_plus_dbl: {v_plus_dbl}")

# def get_impulse_resus(A_decom, v_, v_star, mu, n_cld, d, prev_impulse):
#     # v_: (n_cld*d, 1)
#     n_cld_d = v_.shape[0]
#     f = cp.Variable((n_cld_d, 1))
#     A_decom_p = cp.Parameter(A_decom.shape) # Todo
#     v_p = cp.Parameter((n_cld_d, 1))
#     v_star_p = cp.Parameter((n_cld_d, 1))
#     mu_p = cp.Parameter((mu.shape[0], 1)) 

#     objective = cp.Minimize(0.5 * cp.sum_squares(A_decom_p @ f) + cp.sum(cp.multiply(f, v_p)) - cp.sum(cp.multiply(f, v_star_p)))
#     constraints = [f[i*d] >= 0 for i in range(n_cld)] + \
#                     [cp.SOC(cp.multiply(mu_p[i], f[i*d]), f[i*d+1:i*d+d]) for i in range(n_cld)] +\
#                         [f[0]==prev_impulse[0], f[2]==prev_impulse[2]]

#     problem = cp.Problem(objective, constraints)
#     cvxpylayer = CvxpyLayer(problem, parameters=[A_decom_p, v_p, v_star_p, mu_p], variables=[f])

#     # impulse, = cvxpylayer(A_decom, v_.reshape(-1, 1), torch.zeros_like(v_star.reshape(-1, 1)), mu.reshape(-1, 1))
#     impulse_star, = cvxpylayer(A_decom, v_.reshape(-1, 1), v_star.reshape(-1, 1), mu.reshape(-1, 1))
#     return impulse_star

# impulse_resus = get_impulse_resus(Minv_sqrt_Jac_T, 
#                                     (Jac.reshape(4, 2) @ v_plus).reshape(4, 1),
#                                     torch.zeros_like((Jac.reshape(4, 2) @ v_tch)).reshape(4, 1),
#                                     mu_tch, 2, 2,
#                                     impulse)
# print(f"impulse_resus:{impulse_resus}")
# v_resus = v_plus +  (Minv_Jac_T @ impulse_resus).reshape(2, 1) # special

# print(f"v_resus:{v_resus}")

# add regularizer
# A_sqrt = cp.Parameter((4, 4))
# objective2 = cp.Minimize(0.5 * cp.sum_squares(A_sqrt @ f) + cp.sum(cp.multiply(f, v_)))
# problem2 = cp.Problem(objective2, constraints)

# cvxpylayer2 = CvxpyLayer(problem2, parameters=[A_sqrt, v_, mu], variables=[f])

# A_sqrt_tch = torch.cholesky(A+torch.eye(4, dtype=torch.float32)*1e-2)

# solution2, = cvxpylayer2(A_sqrt_tch, (Jac.reshape(4, 2) @ v_tch).reshape(4, 1), mu_tch) 
# print(solution2)


##############################################
# animation
# from datasets.datasets import RigidBodyDataset
# from systems.bouncing_mass_points import BouncingMassPoints
# from systems.chain_pendulum_with_wall import ChainPendulum_w_Wall
# from pytorch_lightning import seed_everything

# seed_everything(0)

# import matplotlib.pyplot as plt
# plt.switch_backend("TkAgg")

# body = BouncingMassPoints(n_balls=2, m=0.1, l=0.1, mu=0.0, cor=0.0)
# body = ChainPendulum_w_Wall(n_links=1, m=1, l=2.0, mu=0.1, lb=1.1, rb=1.1)
# dataset = RigidBodyDataset(body=body, n_traj=20, chunk_len=200, regen=True)

# plt.plot(dataset.zs[0, :, 1, 0, 1])
# plt.show()

# ani = body.animate(dataset.zs, 7)
# ani.save(f"test.gif")

#################################33

# import torch

# a = torch.tensor([[1,2,3], [4,5,6]])

# def view_tensor(x):
#     # x = x.contiguous()
#     x = x.reshape(3, 2)[0]
#     return x

# a_new = view_tensor(a)

# print(a)

# print(a_new)

