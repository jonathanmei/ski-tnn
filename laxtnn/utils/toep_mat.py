import torch
import torch.nn.functional as F

def torch_toeplitz(col, row_rev):
    """
    Make a dense torch.Tensor Toeplitz matrix from its first col and reversed first row
    col starts with main diagonal, top to bottom
    row_rev ends with 1st off diagonal, right to left
    """
    vals = torch.cat((col, row_rev), axis=-1)
    shape = col.shape[-1], row_rev.shape[-1]+1
    i, j = torch.ones(*shape).nonzero().T
    shape = vals.shape[:-1] + shape
    return vals[..., i-j].reshape(*shape)

class ToepMat:
    """
    Batch of Toeplitz matrices of size (..., n, n), represented by up to 2n elements in dim -1
        that allows us to use the torch syntax: A @ X
    e.g.
        a = torch.cat([zero, pos, zero], dim=0)  # (..., r)
            (causal only)
        a = torch.cat([zero, pos, zero, neg], dim=0)  # (..., 2n-1)
            (neg should be flipped as to make circulant)
    """
    def __init__(self, a, n):
        """
        a (..., r)
        """
        self.a = a
        self.n = n

    def __matmul__(self, other):
        """
        other (..., n, s)
        """
        n2 = 2 * self.n
        y = torch.fft.rfft(other, n2, dim=-2)  # (..., 2n, s)
        v = torch.fft.rfft(self.a, n2, dim=-1)[..., None]  # (..., 2n, 1)
        u = v * y
        output = torch.fft.irfft(u, n2, dim=-2)[..., :self.n, :]  # (..., n, s)
        return output

    def to_mat(self):
        padded = F.pad(self.a, (0, self.n*2-1-self.a.shape[-1]))
        out = torch_toeplitz(padded[..., :self.n], padded[..., 1-self.n:])
        return out
