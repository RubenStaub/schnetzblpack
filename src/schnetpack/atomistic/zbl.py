import torch
import torch.nn as nn
import torch.nn.functional as F
import schnetzblpack as szpk
from schnetzblpack.nn.activations import softplus_inverse
from schnetzblpack import Properties
import numpy as np
import os

__all__ = ["ZBLRepulsionEnergy"]

class ZBLRepulsionEnergy(nn.Module):
    def __init__(self, a0=0.5291772105638411, ke=14.399645351950548, distance_provider=szpk.nn.AtomDistances()):
        super().__init__()
        self.distance_provider = distance_provider
        self.a0 = a0
        self.ke = ke
        self.kehalf = ke / 2
        self.register_parameter("_adiv", nn.Parameter(torch.Tensor(1)))
        self.register_parameter("_apow", nn.Parameter(torch.Tensor(1)))
        self.register_parameter("_c1", nn.Parameter(torch.Tensor(1)))
        self.register_parameter("_c2", nn.Parameter(torch.Tensor(1)))
        self.register_parameter("_c3", nn.Parameter(torch.Tensor(1)))
        self.register_parameter("_c4", nn.Parameter(torch.Tensor(1)))
        self.register_parameter("_a1", nn.Parameter(torch.Tensor(1)))
        self.register_parameter("_a2", nn.Parameter(torch.Tensor(1)))
        self.register_parameter("_a3", nn.Parameter(torch.Tensor(1)))
        self.register_parameter("_a4", nn.Parameter(torch.Tensor(1)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self._adiv, softplus_inverse(1 / (0.8854 * self.a0)))
        nn.init.constant_(self._apow, softplus_inverse(0.23))
        nn.init.constant_(self._c1, softplus_inverse(0.18180))
        nn.init.constant_(self._c2, softplus_inverse(0.50990))
        nn.init.constant_(self._c3, softplus_inverse(0.28020))
        nn.init.constant_(self._c4, softplus_inverse(0.02817))
        nn.init.constant_(self._a1, softplus_inverse(3.20000))
        nn.init.constant_(self._a2, softplus_inverse(0.94230))
        nn.init.constant_(self._a3, softplus_inverse(0.40280))
        nn.init.constant_(self._a4, softplus_inverse(0.20160))

    def forward(self, inputs, distances=None):
        neighbors = inputs[Properties.neighbors]
        neighbor_mask = inputs[Properties.neighbor_mask]
        n_batch, n_atoms, n_neigh = neighbors.shape
        Zf = inputs["_atomic_numbers"].float().unsqueeze(-1)
        if distances is None:
            distances = self.distance_provider(
                inputs[Properties.R],
                neighbors,
                inputs[Properties.cell],
                inputs[Properties.cell_offset],
                neighbor_mask=neighbor_mask,
            )
        r_ij = distances
        z_ex = Zf.expand(n_batch, n_atoms, n_atoms)

        # calculate parameters
        z = z_ex ** F.softplus(self._apow)
        a = (z + z.transpose(1, 2)) * F.softplus(self._adiv)
        # remove diag
        a = torch.gather(a, -1, neighbors) * neighbor_mask

        a1 = F.softplus(self._a1) * a
        a2 = F.softplus(self._a2) * a
        a3 = F.softplus(self._a3) * a
        a4 = F.softplus(self._a4) * a
        c1 = F.softplus(self._c1)
        c2 = F.softplus(self._c2)
        c3 = F.softplus(self._c3)
        c4 = F.softplus(self._c4)
        # normalize c coefficients (to get asymptotically correct behaviour for r -> 0)
        csum = c1 + c2 + c3 + c4
        c1 = c1 / csum
        c2 = c2 / csum
        c3 = c3 / csum
        c4 = c4 / csum
        # actual interactions
        zizj = z_ex * z_ex.transpose(1, 2)
        zizj = torch.gather(zizj, -1, neighbors) * neighbor_mask

        f = (
            c1 * torch.exp(-a1 * r_ij)
            + c2 * torch.exp(-a2 * r_ij)
            + c3 * torch.exp(-a3 * r_ij)
            + c4 * torch.exp(-a4 * r_ij)
        )

        # compute ij values
        corr_ij = torch.where(
            neighbor_mask != 0, self.kehalf * f * zizj / r_ij, torch.zeros_like(r_ij)
        )
        return torch.sum(corr_ij, -1, keepdim=True)
