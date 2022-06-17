
from .mixin import Loss


class SSDLoss(Loss):

    def forward(self, pred, gt):
        B   = gt.size(0)
        loss = (gt - pred).pow(2).view(B, -1).sum()

        return loss, {'loss': loss.detach()}
