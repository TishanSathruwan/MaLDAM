"""
 VICRegL Model Definition 
 Reference: https://arxiv.org/pdf/2210.01571
"""

import torch.nn as nn
from src.utils.vicregl.utils import MLP

class VICRegLHead(nn.Module):
    """
     VICRegL Head Implementation for Malaria 
    """
    def __init__(self, rep_dim=768, maps_mlp="256-128-256", mlp="256-128", norm_layer="layer_norm", alpha=0.75):
        super().__init__()

        self.representation_dim = rep_dim
        self.alpha = alpha

        self.avgpool =  nn.AdaptiveAvgPool2d((1, 1))

        if self.alpha < 1.0:
            self.maps_projector = MLP(maps_mlp, self.representation_dim, norm_layer)

        if self.alpha > 0.0:
            self.projector = MLP(mlp, self.representation_dim, norm_layer)

    def forward(self, feat_maps):
        """ forward pass for VICRegL model """

        outputs = {
            "representation": [],
            "embedding": [],
            "maps_embedding": [],
        }
        for maps in feat_maps:
            # reshape B, C, H, W -> B, HxW, C
            reshaped_maps = maps.flatten(2, 3).permute(0, 2, 1)
            # get representation
            representation = self.avgpool(maps).flatten(1)
            outputs["representation"].append(representation)

            if self.alpha > 0.0:
                embedding = self.projector(representation)
                outputs["embedding"].append(embedding)

            if self.alpha < 1.0:
                batch_size, num_loc, _ = reshaped_maps.shape
                maps_embedding = self.maps_projector(reshaped_maps.flatten(start_dim=0, end_dim=1))
                maps_embedding = maps_embedding.view(batch_size, num_loc, -1)
                outputs["maps_embedding"].append(maps_embedding)

        return outputs

