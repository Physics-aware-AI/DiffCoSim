import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

import torch
from torch.autograd.gradcheck import (
    get_analytical_jacobian, 
    get_numerical_jacobian, 
    _as_tuple, 
    is_tensor_like,
    _differentiable_outputs,
    iter_tensors
)
from symeig import symeig_proj, symeig1, symeig2

def mygradcheck(
    func,  # See Note [VarArg of Tensors]
    inputs,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    raise_exception: bool = True,
    check_sparse_nnz: bool = False,
    nondet_tol: float = 0.0,
    check_undefined_grad: bool = True,
    check_grad_dtypes: bool = False
):
    r"""Check gradients computed via small finite differences against analytical
    gradients w.r.t. tensors in :attr:`inputs` that are of floating point or complex type
    and with ``requires_grad=True``.

    The check between numerical and analytical gradients uses :func:`~torch.allclose`.

    For complex functions, no notion of Jacobian exists. Gradcheck verifies if the numerical and
    analytical values of Wirtinger and Conjugate Wirtinger derivative are consistent. The gradient
    computation is done under the assumption that the overall function has a real valued output.
    For functions with complex output, gradcheck compares the numerical and analytical gradients
    for two values of :attr:`grad_output`: 1 and 1j. For more details, check out
    :ref:`complex_autograd-doc`.

    .. note::
        The default values are designed for :attr:`input` of double precision.
        This check will likely fail if :attr:`input` is of less precision, e.g.,
        ``FloatTensor``.

    .. warning::
       If any checked tensor in :attr:`input` has overlapping memory, i.e.,
       different indices pointing to the same memory address (e.g., from
       :func:`torch.expand`), this check will likely fail because the numerical
       gradients computed by point perturbation at such indices will change
       values at all other indices that share the same memory address.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor or a tuple of Tensors
        inputs (tuple of Tensor or Tensor): inputs to the function
        eps (float, optional): perturbation for finite differences
        atol (float, optional): absolute tolerance
        rtol (float, optional): relative tolerance
        raise_exception (bool, optional): indicating whether to raise an exception if
            the check fails. The exception gives more information about the
            exact nature of the failure. This is helpful when debugging gradchecks.
        check_sparse_nnz (bool, optional): if True, gradcheck allows for SparseTensor input,
            and for any SparseTensor at input, gradcheck will perform check at nnz positions only.
        nondet_tol (float, optional): tolerance for non-determinism. When running
            identical inputs through the differentiation, the results must either match
            exactly (default, 0.0) or be within this tolerance.
        check_undefined_grad (bool, options): if True, check if undefined output grads
            are supported and treated as zeros

    Returns:
        True if all differences satisfy allclose condition
    """
    def fail_test(msg):
        if raise_exception:
            raise RuntimeError(msg)
        return False

    tupled_inputs = _as_tuple(inputs)
    if not check_sparse_nnz and any(t.is_sparse for t in tupled_inputs if isinstance(t, torch.Tensor)):
        return fail_test('gradcheck expects all tensor inputs are dense when check_sparse_nnz is set to False.')

    # Make sure that gradients are saved for at least one input
    any_input_requiring_grad = False
    for idx, inp in enumerate(tupled_inputs):
        if is_tensor_like(inp) and inp.requires_grad:
            if not (inp.dtype == torch.float64 or inp.dtype == torch.complex128):
                warnings.warn(
                    'The {}th input requires gradient and '
                    'is not a double precision floating point or complex. '
                    'This check will likely fail if all the inputs are '
                    'not of double precision floating point or complex. ')
            content = inp._values() if inp.is_sparse else inp
            # TODO: To cover more problematic cases, replace stride = 0 check with
            # "any overlap in memory" once we have a proper function to check it.
            if content.layout is not torch._mkldnn:  # type: ignore
                if not all(st > 0 or sz <= 1 for st, sz in zip(content.stride(), content.size())):
                    raise RuntimeError(
                        'The {}th input has a dimension with stride 0. gradcheck only '
                        'supports inputs that are non-overlapping to be able to '
                        'compute the numerical gradients correctly. You should call '
                        '.contiguous on the input before passing it to gradcheck.')
            any_input_requiring_grad = True
            inp.retain_grad()
    if not any_input_requiring_grad:
        raise ValueError(
            'gradcheck expects at least one input tensor to require gradient, '
            'but none of the them have requires_grad=True.')

    func_out = func(*tupled_inputs)
    output = _differentiable_outputs(func_out)

    if not output:
        for i, o in enumerate(func_out):
            def fn(input):
                return _as_tuple(func(*input))[i]
            numerical = get_numerical_jacobian(fn, tupled_inputs, eps=eps)
            for n in numerical:
                if torch.ne(n, 0).sum() > 0:
                    return fail_test('Numerical gradient for function expected to be zero')
        return True

    a_list = [] ; n_list = []
    for i, o in enumerate(output):
        if not o.requires_grad:
            continue

        def fn(input):
            return _as_tuple(func(*input))[i]

        analytical, reentrant, correct_grad_sizes, correct_grad_types = get_analytical_jacobian(tupled_inputs,
                                                                                                o,
                                                                                                nondet_tol=nondet_tol)
        numerical = get_numerical_jacobian(fn, tupled_inputs, eps=eps)

        out_is_complex = o.is_complex()

        if out_is_complex:
            # analytical vjp with grad_out = 1.0j
            analytical_with_imag_grad_out, reentrant_with_imag_grad_out, \
                correct_grad_sizes_with_imag_grad_out, correct_grad_types_with_imag_grad_out \
                = get_analytical_jacobian(tupled_inputs, o, nondet_tol=nondet_tol, grad_out=1j)
            numerical_with_imag_grad_out = get_numerical_jacobian(fn, tupled_inputs, eps=eps, grad_out=1j)

        if not correct_grad_types and check_grad_dtypes:
            return fail_test('Gradient has dtype mismatch')

        if out_is_complex and not correct_grad_types_with_imag_grad_out and check_grad_dtypes:
            return fail_test('Gradient (calculated using complex valued grad output) has dtype mismatch')

        if not correct_grad_sizes:
            return fail_test('Analytical gradient has incorrect size')

        if out_is_complex and not correct_grad_sizes_with_imag_grad_out:
            return fail_test('Analytical gradient (calculated using complex valued grad output) has incorrect size')

        def checkIfNumericalAnalyticAreClose(a, n, j, error_str=''):
            if not torch.allclose(a, n, rtol, atol):
                return fail_test(error_str + 'Jacobian mismatch for output %d with respect to input %d,\n'
                                 'numerical:%s\nanalytical:%s\n' % (i, j, n, a))

        inp_tensors = iter_tensors(tupled_inputs, True)

        for j, (a, n, inp) in enumerate(zip(analytical, numerical, inp_tensors)):
            if a.numel() != 0 or n.numel() != 0:
                if o.is_complex():
                    # C -> C, R -> C
                    a_with_imag_grad_out = analytical_with_imag_grad_out[j]
                    n_with_imag_grad_out = numerical_with_imag_grad_out[j]
                    checkIfNumericalAnalyticAreClose(a_with_imag_grad_out, n_with_imag_grad_out, j,
                                                     "Gradients failed to compare equal for grad output = 1j. ")
                if inp.is_complex():
                    # C -> R, C -> C
                    checkIfNumericalAnalyticAreClose(a, n, j,
                                                     "Gradients failed to compare equal for grad output = 1. ")
                else:
                    # R -> R, R -> C
                    # checkIfNumericalAnalyticAreClose(a, n, j)
                    a_list.append(a) ; n_list.append(n)

    return a_list, n_list


# def test_original():
#     n = 2
#     a = torch.randn(n, n, dtype=torch.double, requires_grad=True)
#     a0 = a + a.t()
#     a1 = a0.clone()
#     e0, v0 = torch.symeig(a0, eigenvectors=True)
#     e1, v1 = symeig(a1)
#     # a2 = a0.clone()
#     # e2, v2 = symeig(a2)
#     # analytical0, *_ = get_analytical_jacobian((a0,),
#     #                                             v0,
#     #                                             nondet_tol=0.0)
#     # numerical0, *_ = get_numerical_jacobian(lambda x: torch.symeig(x, eigenvectors=True)[1], 
#     #                                         a0)
#     # analytical1, *_ = get_analytical_jacobian((a1,),
#     #                                             v1,
#     #                                             nondet_tol=0.0)
#     # numerical1, *_ = get_numerical_jacobian(a1, v1)
#     # analytical2, *_ = get_analytical_jacobian((a2,),
#     #                                             v2,
#     #                                             nondet_tol=0.0)

#     # assert torch.allclose(analytical0[0], analytical1[0])
#     # assert torch.allclose(analytical0[0], analytical2[0])
    
#     analytical0, numerical0 = mygradcheck(lambda x: torch.symeig(x, eigenvectors=True), a0)
#     analytical1, numerical1 = mygradcheck(lambda x: symeig(x), a1)
#     print(analytical0[1])
#     print(numerical0[1])
#     print(analytical1[1])
#     print(numerical1[1])

def test_proj():
    a = torch.Tensor([[1, 0.001, 0.001],
                    [0.001, 1, 0.001],
                    [0.001, 0.001, 0]])
    a = a.to(dtype=torch.float64)
    a.requires_grad = True
    e, V = symeig_proj(a)
    analytical1, numerical1 = mygradcheck(lambda x: symeig1(x), a)
    print(analytical1[0])
    print(numerical1[0])

if __name__ == "__main__":
    torch.set_printoptions(linewidth=160)
    test_proj()