# -*- coding: utf-8 -*-
import torch
import numpy as np
import scipy

class MatrixSquareRoot(torch.autograd.Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    Taken from: https://github.com/steveli/pytorch-sqrtm/blob/master/sqrtm.py
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input
    
class MatrixSquareRootT(torch.autograd.Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    Taken from: https://github.com/steveli/pytorch-sqrtm/blob/master/sqrtm.py
    """
    @staticmethod
    def forward(ctx, input):
        sqrtm=[]
        for i in range(input.size()[0]):
            m = input[i].detach().cpu().numpy().astype(np.float_)
            sqrtm.append(torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input[i]))
        ctx.save_for_backward(torch.stack(sqrtm))
        return torch.stack(sqrtm)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            tmp = []
            for i in range(sqrtm.shape[0]):
                grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm[i], sqrtm[i], gm[i])
                tmp.append(torch.from_numpy(grad_sqrtm).to(grad_output[i]))
            grad_input = torch.stack(tmp)
        return grad_input