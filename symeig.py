from torch.autograd import Function
import torch

# class Symeig(Function):

#     @staticmethod
#     def forward(ctx, input):
#         lambda_, v = torch.symeig(input, eigenvectors=True, upper=True)
#         ctx.save_for_backward(input, lambda_, v)
#         return lambda_, v

#     @staticmethod
#     def backward(ctx, glambda_, gv):
#         # unpack and initializaiton
#         input, lambda_, v = ctx.saved_tensors
#         result = torch.zeros_like(input)
#         #
#         vh = v.conj().transpose(-2, -1)
#         # contribution from the eigenvectors
#         if gv is not None:
#             F = lambda_.unsqueeze(-2) - lambda_.unsqueeze(-1)
#             F.diagonal(0, -2, -1).fill_(float("Inf"))
#             F.pow_(-1)
#             result = v @ ((F * (vh @ gv)) @ vh)
#         else:
#             result = torch.zeros_like(input)
#         # contribution from eigenvalues
#         if glambda_ is not None:
#             glambda_ = glambda_.type_as(input)
#             glambda_term = (v * glambda_.unsqueeze(-2)) @ vh
#             result = result + glambda_term

#         grad_input = result.add(result.conj().transpose(-2, -1)).mul_(0.5)

#         return grad_input

# symeig = Symeig.apply


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
            # F = lambda_.unsqueeze(-2) - lambda_.unsqueeze(-1)
            # F.diagonal(0, -2, -1).fill_(float("Inf"))
            # F.pow_(-1)
            # result = v @ ((F * (vh @ gv)) @ vh)
            gv_term = (v * glambda_.unsqueeze(-2)) @ gv
            result = gv_term + gv_term.transpose(-2, -1)
        else:
            result = torch.zeros_like(input)
        # contribution from eigenvalues
        if glambda_ is not None:
            glambda_ = glambda_.type_as(input)
            glambda_term = (v * glambda_.unsqueeze(-2)) @ vh
            result = result + glambda_term

        grad_input = result.add(result.conj().transpose(-2, -1)).mul_(0.5)

        return grad_input

symeig = Symeig.apply