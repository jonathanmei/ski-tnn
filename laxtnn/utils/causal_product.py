import torch

from .toep_mat import ToepMat

def causal_product_naive_cumsum(q, k, v, chunk=1):
    # From: https://github.com/lucidrains/performer-pytorch/blob/main/performer_pytorch/performer_pytorch.py
    # inefficient causal linear attention, without cuda code, for reader's reference
    # chunk size determines the amount of parallelism. 1 seems to be optimal...

    last_context_cumsum = 0
    outs = []

    for q, k, v in zip(*map(lambda t: t.chunk(chunk, dim=-2), (q, k, v))):
        context = torch.einsum('...nd,...ne->...nde', k, v)
        context_cumsum = last_context_cumsum + context.cumsum(dim=-3)
        out = torch.einsum('...nde,...nd->...ne', context_cumsum, q)

        last_context_cumsum = context_cumsum[..., -1:, :, :]
        outs.append(out)

    return torch.cat(outs, dim=-2)

# def causal_product_trio(U, S, V, X):
#     """
#     U, V (..., n, r)  require_grad == `False`
#     S (..., r, r)  (may be either a Tensor or an object that applies the action of matrix)
#     X (..., n)
#     This version will use too much memory!
#     """
#     VX = V * X[..., None]  # (..., n, r)
#     VX_cumsum = VX.cumsum(dim=-2)  # (..., n, r)
#     SVX_cumsum = S @ VX_cumsum.transpose(-1, -2)  # (..., r, n)
#     Y = torch.einsum('...rn, ...nr -> ...n', SVX_cumsum, U)
#     return Y

class CausalProductTrio(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, S, V, X):
        """
        U, V (..., n, r)  require_grad == `False`
        S (..., r, r)
        X (..., n)
        """
        VX = V * X[..., None]  # (..., n, r)
        VX_cumsum = VX.cumsum(dim=-2)  # (..., n, r)
        ctx.save_for_backward(U, S, V, VX_cumsum)
        SVX_cumsum = S @ VX_cumsum.transpose(-1,-2)  # (..., r, n)
        Y = torch.einsum('...rn, ...nr -> ...n', SVX_cumsum, U)
        return Y

    @staticmethod
    def backward(ctx, dY):
        """
        dY (..., n)
        U, V (..., n, r)  require_grad == `False`
        S (..., r, r)
        VX_cumsum (..., n, r)  saved from fw
        """
        U, S, V, VX_cumsum = ctx.saved_tensors
        UdY = U * dY[..., None]  # (..., n, r)
        dS = UdY.transpose(-1, -2) @ VX_cumsum  # (..., r, r)
        UdY_cumsum = UdY.cumsum(dim=-2)  # (..., n, r)
        # invert the cumsum:
        UdY_cumsum = UdY_cumsum[..., -1:, :] + UdY - UdY_cumsum   # (..., n, r)
        SVt = S @ V.transpose(-1, -2)  # (..., r, n)
        dX = torch.einsum('...nr, ...rn -> ...n', UdY_cumsum, SVt)
        return None, dS, None, dX

class CausalProductTrioToep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, a, X):
        """
        U (..., n, r)  require_grad == `False`
        V = U
        a (..., s)  the up to 2r-1 entries to represent a Toep matrix
        X (..., n)
        """
        r = U.shape[-1]
        VX = U * X[..., None]  # (..., n, r)
        VX_cumsum = VX.cumsum(dim=-2)  # (..., n, r)
        ctx.save_for_backward(U, a, VX_cumsum)
        S = ToepMat(a, r)  # (..., r, r)
        SVX_cumsum = S @ VX_cumsum.transpose(-1,-2)  # (..., r, n)
        Y = torch.einsum('...rn, ...nr -> ...n', SVX_cumsum, U)
        return Y

    @staticmethod
    def backward(ctx, dY):
        """
        dY (..., n)
        U (..., n, r)  require_grad == `False`
        a (..., s)
        VX_cumsum (..., n, r)  saved from fw
        """
        U, a, VX_cumsum = ctx.saved_tensors
        r = U.shape[-1]
        s = a.shape[-1]
        UdY = U * dY[..., None]  # (..., n, r)
        dS = UdY.transpose(-1, -2) @ VX_cumsum  # (..., r, r)
        pos = [torch.diagonal(dS, offset=i).sum(dim=-1) for i in [0] + [-j for j in range(r)]]
        neg = [torch.diagonal(dS, offset=i).sum(dim=-1) for i in range(r-1, 0, -1)]
        da = torch.stack((pos+neg)[:s], dim=-1)  # (..., s)
        UdY_cumsum = UdY.cumsum(dim=-2)  # (..., n, r)
        # invert the cumsum:
        UdY_cumsum = UdY_cumsum[..., -1:, :] + UdY - UdY_cumsum   # (..., n, r)
        S = ToepMat(a, r)
        SVt = S @ U.transpose(-1, -2)  # (..., r, n)
        dX = torch.einsum('...nr, ...rn -> ...n', UdY_cumsum, SVt)
        return None, da, None, dX

class CausalProductTrioRot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, a, c, s, X):
        """
        r == 2k+1
        U (..., n, r)  require_grad == `False`
        V = U
        a (..., k+1) the magnitudes of the cosines and const
        c, s (..., k)  the 2 entries (cos(phi), sin(phi)) to represent a block diag of (2x2 blocks) rotation matrix
        X (..., n)
        """
        VX = U * X[..., None]  # (..., n, r)
        VX_cumsum = VX.cumsum(dim=-2)  # (..., n, r)
        ctx.save_for_backward(U, a, c, s, VX_cumsum)
        # apply rot mat sparsely
        x1, x2 = VX_cumsum[..., 0::2], VX_cumsum[..., 1:-1:2]
        cos, sin = c[..., None, :], s[..., None, :]
        a1 = a[..., None, :-1]
        x0 = VX_cumsum[..., -1:]
        a0 = a[..., None, -1:]
        SVX_cumsum = torch.stack([a1 *(x1 * cos - x2 * sin), a1 * (x2 * cos + x1 * sin)], dim=-1).flatten(-2, -1)  # (..., r, n)
        SVX_cumsum = torch.cat([SVX_cumsum, a0 * x0], dim=-1)  # (..., n, r)
        Y = torch.einsum('...nr, ...nr -> ...n', SVX_cumsum, U)
        return Y

    @staticmethod
    def backward(ctx, dY):
        """
        dY (..., n)
        U (..., n, r)  require_grad == `False`
        a (..., k+1)
        c, s (..., k)
        VX_cumsum (..., n, 2k+1)  saved from fw
        """
        U, a, c, s, VX_cumsum = ctx.saved_tensors
        UdY = U * dY[..., None]  # (..., n, r)
        dS = UdY.transpose(-1, -2) @ VX_cumsum  # (..., r, r)
        # TODO: rotmat stuff
        UdY_cumsum = UdY.cumsum(dim=-2)  # (..., n, r)
        # invert the cumsum:
        UdY_cumsum = UdY_cumsum[..., -1:, :] + UdY - UdY_cumsum   # (..., n, r)
        # apply rotmat sparsely
        x1, x2 = U[..., 0::2], U[..., 1:-1:2]
        cos, sin = c[..., None, :], s[..., None, :]
        a1 = a[..., None, :-1]
        x0 = VX_cumsum[..., -1:]
        a0 = a[..., None, -1:]
        SVt = torch.stack([a1 *(x1 * cos - x2 * sin), a1 * (x2 * cos + x1 * sin)], dim=-1).flatten(-2, -1)  # (..., r, n)
        SVt = torch.cat([SVt, a0 * x0], dim=-1)  # (..., n, r)
        dX = torch.einsum('...nr, ...nr -> ...n', UdY_cumsum, SVt)
        return None, da, dc, ds, None, dX


causal_product_trio = CausalProductTrio.apply
causal_product_trio_toep = CausalProductTrioToep.apply
causal_product_trio_rot = CausalProductTrioRot.apply