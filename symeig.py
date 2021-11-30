"""
code from https://colab.research.google.com/drive/1wyt3357g8J91PQggLh8CLgvm4RiBmjQv
please see https://github.com/pytorch/pytorch/issues/47599
"""

from torch.autograd import Function
import torch


class Symeig(Function):

    @staticmethod
    def forward(ctx, input):
        lambda_, v = torch.symeig(input, eigenvectors=True, upper=True)
        ctx.save_for_backward(input, lambda_, v)
        return lambda_, v

    @staticmethod
    def backward(ctx, glambda_, gv):
        # unpack and initializaiton
        input, lambda_, v = ctx.saved_tensors
        grad_input = None
        #
        vh = v.conj().transpose(-2, -1)
        # contribution from the eigenvectors
        if gv is not None:
            F = lambda_.unsqueeze(-2) - lambda_.unsqueeze(-1)
            # F.diagonal(0, -2, -1).fill_(float("Inf"))
            # idx = (lambda_ < 0.5).to(dtype=torch.int).sum()
            # assert torch.allclose(lambda_[:idx], torch.zeros(idx).type_as(lambda_), atol=1e-6)
            # assert torch.allclose(lambda_[idx:], torch.ones(idx).type_as(lambda_), atol=1e-6)
            # F[..., :idx, :idx].fill_(float("Inf"))
            # F[..., idx:, idx:].fill_(float("Inf"))
            min_threshold = 1e-6
            idx = torch.abs(F) < min_threshold
            F[idx] = float("inf")

            idx = torch.abs(F) < min_threshold
            Fsign = torch.sign(F[idx])
            F[idx] = Fsign * min_threshold

            F.pow_(-1)
            result = v @ ((F * (vh @ gv)) @ vh)
            # gv_term = (v * lambda_.unsqueeze(-2)) @ gv.conj().transpose(-2, -1)
            # result = gv_term + gv_term.transpose(-2, -1)
        else:
            result = torch.zeros_like(input)
        # contribution from eigenvalues
        if glambda_ is not None:
            glambda_ = glambda_.type_as(input)
            glambda_term = (v * glambda_.unsqueeze(-2)) @ vh
            result = result + glambda_term

        grad_input = result.add(result.conj().transpose(-2, -1)).mul_(0.5)

        return grad_input

class symeig1_fcn(torch.autograd.Function):
  @staticmethod
  def forward(ctx, A):
    eival, eivec = torch.symeig(A, eigenvectors=True)
    ctx.save_for_backward(eival, eivec)
    return eival, eivec
  
  @staticmethod
  def backward(ctx, grad_eival, grad_eivec):
    # parameters to adjust
    min_threshold = 1e-6

    eival, eivec = ctx.saved_tensors
    eivect = eivec.transpose(-2, -1)

    # for demo only: only take the contribution from grad_eivec
    F = eival.unsqueeze(-2) - eival.unsqueeze(-1)

    # modified step: change the difference of degenerate eigenvalues with inf,
    # instead of only changing the diagonal
    idx = torch.abs(F) < min_threshold
    F[idx] = float("inf")

    # an additional step: clip the value to have min_threshold so the 
    # instability isn't severe
    idx = torch.abs(F) < min_threshold
    Fsign = torch.sign(F[idx])
    F[idx] = Fsign * min_threshold

    F = F.pow(-1)
    F = F * (eivect @ grad_eivec)
    res = eivec @ F @ eivect
    return (res + res.transpose(-2, -1)) * 0.5, None

class symeig2_fcn(torch.autograd.Function):
  @staticmethod
  def forward(ctx, A):
    eival, eivec = torch.symeig(A, eigenvectors=True)
    ctx.save_for_backward(eival, eivec)
    return eival, eivec
  
  @staticmethod
  def backward(ctx, grad_eival, grad_eivec):
    # parameters to adjust
    min_threshold = 1e-6

    eival, eivec = ctx.saved_tensors
    eivect = eivec.transpose(-2, -1)

    # for demo only: only take the contribution from grad_eivec
    F = eival.unsqueeze(-2) - eival.unsqueeze(-1)
    Fdiag = F.diagonal()
    Fdiag[:] = float("inf")

    # modified step as in symeig1
    idx = torch.abs(F) < min_threshold
    F[idx] = float("inf")

    # additional step: check the condition and return `nan` if not satisfied
    degenerate = torch.any(idx)
    if degenerate:
      # check if the loss function does not depend on which linear combination
      # of the degenerate eigenvectors
      # (ref: https://arxiv.org/pdf/2011.04366.pdf eq. 2.13)
      xtg = eivect @ grad_eivec
      diff_xtg = (xtg - xtg.transpose(-2, -1))[idx]
      reqsat = torch.allclose(diff_xtg, torch.zeros_like(diff_xtg))
      # if the requirement is not satisfied, mathematically the derivative
      # should be `nan`.
      if not reqsat:
        res = torch.zeros_like(eivec) + float("nan")
        return res, None

    F = F.pow(-1)
    F = F * (eivect @ grad_eivec)
    res = eivec @ F @ eivect
    return (res + res.transpose(-2, -1)) * 0.5, None

symeig1 = symeig1_fcn.apply
symeig2 = symeig2_fcn.apply

symeig_proj = Symeig.apply