import math

import torch
from torch import nn
from torch.autograd.function import Function

'''
Variable projection transformation using adaptive Hermite functions.
Here the orthogonal projection is passed.
'''
class vpfun_apr(Function):
    """
    Implementation based on

    [1] P. Kovács, G. Bognár, C. Huber and M. Huemer, VPNet - Variable Projection Network
    (2021, [Online]), Available: https://git.siliconaustria.com/pub/sparseestimation/vpnet
    """
    @staticmethod
    def forward(ctx, x, params, ada, device, penalty):
        ctx.device = device
        ctx.penalty = penalty
        phi, dphi, ind = ada(params)
        phip = torch.linalg.pinv(phi)
        coeffs = phip @ torch.transpose(x, 1, 2)
        y_est = torch.transpose(phi @ coeffs, 1, 2)
        nparams = torch.tensor(max(params.shape))
        ctx.save_for_backward(x, phi, phip, dphi, ind, coeffs, y_est, nparams)

        return y_est

    @staticmethod
    def backward(ctx, dy):
        x, phi, phip, dphi, ind, coeffs, y_est, nparams = ctx.saved_tensors
        #dx = dy @ phip
        dx = (dy @ phi) @ phip
        dp = None
        wdphi_r = (x - y_est) @ dphi
        phipc = torch.transpose(phip, -1, -2) @ coeffs  # (N,L,C)

        batch = x.shape[0]
        t2 = torch.zeros(
            batch, 1, phi.shape[1], nparams, dtype=x.dtype, device=ctx.device)
        jac1 = torch.zeros(
            batch, 1, phi.shape[0], nparams, dtype=x.dtype, device=ctx.device)
        jac3 = torch.zeros(
            batch, 1, phi.shape[1], nparams, dtype=x.dtype, device=ctx.device)
        for j in range(nparams):
            rng = ind[1, :] == j
            indrows = ind[0, rng]
            jac1[:, :, :, j] = torch.transpose(dphi[:, rng] @ coeffs[:, indrows, :], 1, 2)  # (N,C,L)
            t2[:, :, indrows, j] = wdphi_r[:, :, rng]
            jac3[:, :, indrows, j] = torch.transpose(phipc, 1, 2) @ dphi[:, rng]

        jac = jac1 - phi @ (phip @ jac1) + torch.transpose(phip, -1, -2) @ t2

        dy = dy.unsqueeze(-1)
        res = (x - y_est) / (x ** 2).sum(dim=2, keepdim=True)
        res = res.unsqueeze(-1)
        dp = (jac * dy).mean(dim=0).sum(dim=1) - 2 * \
            ctx.penalty * (jac1 * res).mean(dim=0).sum(dim=1)

        return dx, dp, None, None, None


'''
Variable projection layer with the projection being passed.
'''
class vp_layer_apr(nn.Module):
    """
    Implementation based on

    [1] P. Kovács, G. Bognár, C. Huber and M. Huemer, VPNet - Variable Projection Network
    (2021, [Online]), Available: https://git.siliconaustria.com/pub/sparseestimation/vpnet
    """
    def __init__(self, ada, n_in, n_out, nparams, penalty=0.0, dtype=torch.float, device=None, init=None):
        if init is None:
            init = [0.1, 0.0]
        super().__init__()
        self.device = device
        self.n_in = n_in
        self.n_out = n_out
        self.nparams = nparams
        self.penalty = penalty
        self.ada = lambda params: ada(n_in, n_out, params, dtype=dtype, device=self.device)
        self.weight = nn.Parameter(torch.tensor(init))

    def forward(self, input):
        return vpfun_apr.apply(input, self.weight, self.ada, self.device, self.penalty)


'''
Generate Hermite functions (equidistant sampling)
'''
def ada_hermite(m, n, params, dtype=torch.float, device=None):
    """
    Implementation based on

    [1] P. Kovács, G. Bognár, C. Huber and M. Huemer, VPNet - Variable Projection Network
    (2021, [Online]), Available: https://git.siliconaustria.com/pub/sparseestimation/vpnet
    """

    dilation, translation = params[:2]
    t = torch.arange(-(m // 2), m // 2 + 1, dtype=dtype) if m % 2 else torch.arange(-(m / 2), m / 2, dtype=dtype,
                                                                                    device=device)
    x = dilation * (t - translation * m / 2)
    w = torch.exp(-0.5 * x ** 2)
    dw = -x * w
    pi_sqrt = torch.sqrt(torch.sqrt(torch.tensor(math.pi, device=device)))

    # Phi, dPhi
    Phi = torch.zeros((m, n), dtype=dtype, device=device)
    Phi[:, 0] = 1
    Phi[:, 1] = 2 * x
    for j in range(1, n - 1):
        Phi[:, j + 1] = 2 * (x * Phi[:, j] - j * Phi[:, j - 1])

    Phi[:, 0] = w * Phi[:, 0] / pi_sqrt
    dPhi = torch.zeros(m, 2 * n, dtype=dtype, device=device)
    dPhi[:, 0] = dw / pi_sqrt
    dPhi[:, 1] = dPhi[:, 0]

    f = 1
    for j in range(1, n):
        f *= j
        Phi[:, j] = w * Phi[:, j] / \
            torch.sqrt(torch.tensor(2 ** j * f, dtype=dtype, device=device)) / pi_sqrt
        dPhi[:, 2 * j] = torch.sqrt(torch.tensor(2 * j, dtype=dtype, device=device)) * Phi[:, j - 1] - x * Phi[:, j]
        dPhi[:, 2 * j + 1] = dPhi[:, 2 * j]

    t = t[:, None]
    dPhi[:, 0::2] = dPhi[:, 0::2] * (t - translation * m / 2)
    dPhi[:, 1::2] = -dPhi[:, 1::2] * dilation * m / 2

    # ind
    ind = torch.zeros((2, 2 * n), dtype=torch.int64, device=device)
    ind[0, 0::2] = torch.arange(n, dtype=torch.int64, device=device)
    ind[0, 1::2] = torch.arange(n, dtype=torch.int64, device=device)
    ind[1, 0::2] = torch.zeros((1, n), dtype=torch.int64, device=device)
    ind[1, 1::2] = torch.ones((1, n), dtype=torch.int64, device=device)

    return Phi, dPhi, ind


'''
Variable projection transformation using adaptive Hermite functions.
Here the linear coefficients of the orthogonal projection are passed.
'''
class vpfun_coeffs(Function):
    """
    Implementation based on

    [1] P. Kovács, G. Bognár, C. Huber and M. Huemer, VPNet - Variable Projection Network
    (2021, [Online]), Available: https://git.siliconaustria.com/pub/sparseestimation/vpnet
    """
    @staticmethod
    def forward(ctx, x, params, ada, device, penalty):
        ctx.device = device
        ctx.penalty = penalty
        phi, dphi, ind = ada(params)
        phip = torch.linalg.pinv(phi)
        coeffs = phip @ torch.transpose(x, 1, 2)
        y_est = torch.transpose(phi @ coeffs, 1, 2)
        nparams = torch.tensor(max(params.shape))
        ctx.save_for_backward(x, phi, phip, dphi, ind, coeffs, y_est, nparams)
        return torch.transpose(coeffs, 1, 2)

    @staticmethod
    def backward(ctx, dy):
        x, phi, phip, dphi, ind, coeffs, y_est, nparams = ctx.saved_tensors
        dx = dy @ phip
        dp = None
        wdphi_r = (x - y_est) @ dphi
        phipc = torch.transpose(phip, -1, -2) @ coeffs  # (N,L,C)

        batch = x.shape[0]
        t2 = torch.zeros(
            batch, 1, phi.shape[1], nparams, dtype=x.dtype, device=ctx.device)
        jac1 = torch.zeros(
            batch, 1, phi.shape[0], nparams, dtype=x.dtype, device=ctx.device)
        jac3 = torch.zeros(
            batch, 1, phi.shape[1], nparams, dtype=x.dtype, device=ctx.device)
        for j in range(nparams):
            rng = ind[1, :] == j
            indrows = ind[0, rng]
            jac1[:, :, :, j] = torch.transpose(dphi[:, rng] @ coeffs[:, indrows, :], 1, 2)  # (N,C,L)
            t2[:, :, indrows, j] = wdphi_r[:, :, rng]
            jac3[:, :, indrows, j] = torch.transpose(phipc, 1, 2) @ dphi[:, rng]

        jac = -phip @ jac1 + phip @ (torch.transpose(phip, -1, -2) @ t2) + jac3 - phip @ (phi @ jac3)

        dy = dy.unsqueeze(-1)
        res = (x - y_est) / (x ** 2).sum(dim=2, keepdim=True)
        res = res.unsqueeze(-1)
        dp = (jac * dy).mean(dim=0).sum(dim=1) - 2 * \
            ctx.penalty * (jac1 * res).mean(dim=0).sum(dim=1)

        return dx, dp, None, None, None

'''
Variable projection layer with the linear parameters being passed.
'''
class vp_layer_coeffs(nn.Module):
    """
    Implementation based on

    [1] P. Kovács, G. Bognár, C. Huber and M. Huemer, VPNet - Variable Projection Network
    (2021, [Online]), Available: https://git.siliconaustria.com/pub/sparseestimation/vpnet
    """
    def __init__(self, ada, n_in, n_out, nparams, penalty=0.0, dtype=torch.float, device=None, init=None):
        if init is None:
            init = [0.1, 0.0]
        super().__init__()
        self.device = device
        self.n_in = n_in
        self.n_out = n_out
        self.nparams = nparams
        self.penalty = penalty
        self.ada = lambda params: ada(n_in, n_out, params, dtype=dtype, device=self.device)
        self.weight = nn.Parameter(torch.tensor(init))

    def forward(self, input):
        return vpfun_coeffs.apply(input, self.weight, self.ada, self.device, self.penalty)