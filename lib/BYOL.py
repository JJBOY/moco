import torch
import torch.nn as nn
from .common_ops import ProjectionMLP, PredictionMLP
import copy


class BYOL(nn.Module):
    """
    Build a SimSiam model.
    Reference: [Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566)
    """

    def __init__(self, base_encoder, dim=2048, wrap=False,
                 symmetric=False, projector=False, predictor=False, momentum_encoder=0.99):
        """
        base_encoder: encoder network
        dim: feature dimension (default: 2048)
        wrap: warp net encoder to fit CIFAR datasets (default: false)
        """
        super(BYOL, self).__init__()
        self.m = momentum_encoder
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

        self.encoder_k = copy.deepcopy(self.encoder_q)

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

        z1 = self.encoder_q(im_q)
        q1 = self.predictor(z1)
        z2 = self.encoder_k(im_k)
        loss = -torch.nn.functional.cosine_similarity(q1, z2.detach()).mean()
        if self.symmetric:
            z1_s = self.encoder_q(im_k)
            q1_s = self.predictor(z1_s)
            z2_s = self.encoder_k(im_q)
            loss += -torch.nn.functional.cosine_similarity(q1_s, z2_s.detach()).mean()

        return loss * 0.5

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
