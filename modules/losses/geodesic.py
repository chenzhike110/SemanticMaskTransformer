import torch
import torch.nn as nn

class geodesic_loss_R(nn.Module):
    def __init__(self):
        super(geodesic_loss_R, self).__init__()
        self.eps = 1e-6

    # batch geodesic loss for rotation matrices
    def bgdR(self,m1,m2):
        """
        M: (..., 3, 3)
        """
        m = torch.einsum('...ij,...jk->...ik', m1, m2.transpose(-1, -2))

        traces = m.diagonal(dim1=-2, dim2=-1).sum(-1)
        dists = torch.acos(torch.clamp((traces - 1) / 2, -1 + self.eps, 1 - self.eps))
        return dists

    def forward(self, ypred, ytrue):
        theta = self.bgdR(ypred,ytrue)
        return theta