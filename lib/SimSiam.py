import torch
import torch.nn as nn
from .common_ops import ProjectionMLP, PredictionMLP


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    Reference: [Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566)
    """

    def __init__(self, base_encoder, dim=2048, wrap=False,
                 symmetric=False, projector=False, predictor=False):
        """
        base_encoder: encoder network
        dim: feature dimension (default: 2048)
        wrap: warp net encoder to fit CIFAR datasets (default: false)
        """
        super(SimSiam, self).__init__()

        self.symmetric = symmetric

        # create the encoder
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        if wrap:
            self.encoder_q.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.encoder_q.maxpool = nn.Identity()

        dim_mlp = self.encoder_q.fc.weight.shape[1]
        if projector:  # using 3 layers MLP
            self.encoder_q.fc = ProjectionMLP(dim_mlp, dim, dim)
        else:  # using 2 layers MLP
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
        if predictor:
            self.predictor = PredictionMLP(dim, dim // 4, dim)
        else:
            self.predictor = nn.Identity()

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """

        z1 = self.encoder_q(im_q)
        p1 = self.predictor(z1)
        z2 = self.encoder_q(im_k)
        p2 = self.predictor(z2)
        loss_12 = -torch.nn.functional.cosine_similarity(p2, z1.detach()).mean()
        if not self.symmetric:
            return loss_12
        loss_21 = -torch.nn.functional.cosine_similarity(p1, z2.detach()).mean()
        loss = (loss_12 + loss_21) * 0.5
        return loss

