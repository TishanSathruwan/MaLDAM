""" VICRegL Loss """

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.vicregl.utils import (
    gather_center,
    neirest_neighbores_on_l2,
    off_diagonal,
    batch_all_gather,
)


class VICRegLLoss(nn.Module):
    """ Definition of VICRegL Loss """
    def __init__(
        self,
        inv_coeff=25.0,
        cov_coeff=1.0,
        var_coeff=25.0,
        embedding_dim=128,
        l2_all_matches=False,
        num_matches=[20, 20],
        fast_vc_reg=False,
        alpha=0.75,
        ):
        super().__init__()
        self.inv_coeff = inv_coeff
        self.cov_coeff = cov_coeff
        self.var_coeff = var_coeff
        self.embedding_dim = embedding_dim
        self.l2_all_matches = l2_all_matches
        self.num_matches = num_matches
        self.fast_vc_reg = fast_vc_reg
        self.alpha = alpha

    def _vicreg_loss(self, x, y):
        repr_loss = self.inv_coeff * F.mse_loss(x, y)

        x = gather_center(x)
        y = gather_center(y)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = self.var_coeff * (
            torch.mean(F.relu(1.0 - std_x)) / 2 + torch.mean(F.relu(1.0 - std_y)) / 2
        )

        x = x.permute((1, 0, 2))
        y = y.permute((1, 0, 2))

        *_, sample_size, num_channels = x.shape
        non_diag_mask = ~torch.eye(num_channels, device=x.device, dtype=torch.bool)
        # Center features
        # centered.shape = NC
        x = x - x.mean(dim=-2, keepdim=True)
        y = y - y.mean(dim=-2, keepdim=True)

        cov_x = torch.einsum("...nc,...nd->...cd", x, x) / (sample_size - 1)
        cov_y = torch.einsum("...nc,...nd->...cd", y, y) / (sample_size - 1)
        cov_loss = (cov_x[..., non_diag_mask].pow(2).sum(-1) / num_channels) / 2 + (
            cov_y[..., non_diag_mask].pow(2).sum(-1) / num_channels
        ) / 2
        cov_loss = cov_loss.mean()
        cov_loss = self.cov_coeff * cov_loss

        return repr_loss, std_loss, cov_loss
    
    def _local_loss(
        self, maps_1, maps_2,
        mask=None,
    ):
        inv_loss = 0.0
        var_loss = 0.0
        cov_loss = 0.0

        # L2 distance based bacthing
        if self.l2_all_matches:
            num_matches_on_l2 = [None, None]
        else:
            num_matches_on_l2 = self.num_matches

        maps_1_filtered, maps_1_nn = neirest_neighbores_on_l2(
            maps_1, maps_2, num_matches=num_matches_on_l2[0], mask=mask
        )
        maps_2_filtered, maps_2_nn = neirest_neighbores_on_l2(
            maps_2, maps_1, num_matches=num_matches_on_l2[1], mask=mask
        )

        if self.fast_vc_reg:
            inv_loss_1 = F.mse_loss(maps_1_filtered, maps_1_nn)
            inv_loss_2 = F.mse_loss(maps_2_filtered, maps_2_nn)
        else:
            inv_loss_1, var_loss_1, cov_loss_1 = self._vicreg_loss(maps_1_filtered, maps_1_nn)
            inv_loss_2, var_loss_2, cov_loss_2 = self._vicreg_loss(maps_2_filtered, maps_2_nn)
            var_loss = var_loss + (var_loss_1 / 2 + var_loss_2 / 2)
            cov_loss = cov_loss + (cov_loss_1 / 2 + cov_loss_2 / 2)

        inv_loss = inv_loss + (inv_loss_1 / 2 + inv_loss_2 / 2)

        return inv_loss, var_loss, cov_loss
    
    def local_loss(self, maps_embedding, mask=None):
        num_views = len(maps_embedding)
        inv_loss = 0.0
        var_loss = 0.0
        cov_loss = 0.0
        iter_ = 0
        for i in range(2):
            for j in np.delete(np.arange(np.sum(num_views)), i):
                inv_loss_this, var_loss_this, cov_loss_this = self._local_loss(
                    maps_embedding[i], maps_embedding[j], mask=mask
                )
                inv_loss = inv_loss + inv_loss_this
                var_loss = var_loss + var_loss_this
                cov_loss = cov_loss + cov_loss_this
                iter_ += 1

        if self.fast_vc_reg:
            inv_loss = self.inv_coeff * inv_loss / iter_
            var_loss = 0.0
            cov_loss = 0.0
            iter_ = 0
            for i in range(num_views):
                x = gather_center(maps_embedding[i])
                std_x = torch.sqrt(x.var(dim=0) + 0.0001)
                var_loss = var_loss + torch.mean(torch.relu(1.0 - std_x))
                x = x.permute(1, 0, 2)
                *_, sample_size, num_channels = x.shape
                non_diag_mask = ~torch.eye(num_channels, device=x.device, dtype=torch.bool)
                x = x - x.mean(dim=-2, keepdim=True)
                cov_x = torch.einsum("...nc,...nd->...cd", x, x) / (sample_size - 1)
                cov_loss = cov_x[..., non_diag_mask].pow(2).sum(-1) / num_channels
                cov_loss = cov_loss + cov_loss.mean()
                iter_ = iter_ + 1
            var_loss = self.var_coeff * var_loss / iter_
            cov_loss = self.cov_coeff * cov_loss / iter_
        else:
            inv_loss = inv_loss / iter_
            var_loss = var_loss / iter_
            cov_loss = cov_loss / iter_

        return inv_loss, var_loss, cov_loss

    def global_loss(self, embedding, maps=False):
        num_views = len(embedding)
        inv_loss = 0.0
        iter_ = 0
        for i in range(2):
            for j in np.delete(np.arange(np.sum(num_views)), i):
                inv_loss = inv_loss + F.mse_loss(embedding[i], embedding[j])
                iter_ = iter_ + 1
        inv_loss = self.inv_coeff * inv_loss / iter_

        var_loss = 0.0
        cov_loss = 0.0
        iter_ = 0
        for i in range(num_views):
            x = gather_center(embedding[i])
            std_x = torch.sqrt(x.var(dim=0) + 0.0001)
            var_loss = var_loss + torch.mean(torch.relu(1.0 - std_x))
            cov_x = (x.T @ x) / (x.size(0) - 1)
            cov_loss = cov_loss + off_diagonal(cov_x).pow_(2).sum().div(
                self.embedding_dim
            )
            iter_ = iter_ + 1
        var_loss = self.var_coeff * var_loss / iter_
        cov_loss = self.cov_coeff * cov_loss / iter_

        return inv_loss, var_loss, cov_loss

    def compute_metrics(self, outputs):
        """ compute metrics """
        def correlation_metric(x):
            x_centered = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-05)
            return torch.mean(
                off_diagonal((x_centered.T @ x_centered) / (x.size(0) - 1))
            )

        def std_metric(x):
            x = F.normalize(x, p=2, dim=1)
            return torch.mean(x.std(dim=0))

        representation = batch_all_gather(outputs["representation"][0])
        corr = correlation_metric(representation)
        stdrepr = std_metric(representation)

        if self.alpha > 0.0:
            embedding = batch_all_gather(outputs["embedding"][0])
            core = correlation_metric(embedding)
            stdemb = std_metric(embedding)
            return dict(stdr=stdrepr, stde=stdemb, corr=corr, core=core)

        return dict(stdr=stdrepr, corr=corr)
    
    def forward(self, outputs, mask=None):
        """ forward pass """
        
        # Initiate parameters
        loss = 0.0

        # calculate  metrics
        with torch.no_grad():
            logs = self.compute_metrics(outputs)

        # Global criterion
        if self.alpha > 0.0:
            inv_loss, var_loss, cov_loss = self.global_loss(
                outputs["embedding"]
            )
            loss = loss + self.alpha * (inv_loss + var_loss + cov_loss)
            logs.update(dict(inv_l=inv_loss, var_l=var_loss, cov_l=cov_loss,))

        # Local criterion
        # Maps shape: B, C, H, W
        # With convnext actual maps shape is: B, H * W, C
        if self.alpha < 1.0:
            (
                maps_inv_loss,
                maps_var_loss,
                maps_cov_loss,
            ) = self.local_loss(
                outputs["maps_embedding"], mask=mask
            )
            loss = loss + (1 - self.alpha) * (
                maps_inv_loss + maps_var_loss + maps_cov_loss
            )
            logs.update(
                dict(minv_l=maps_inv_loss, mvar_l=maps_var_loss, mcov_l=maps_cov_loss,)
            )

        return loss, logs
