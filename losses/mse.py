
from .mixin import Loss


class MSELoss(Loss):

    def forward(self, pred, gt):
        B   = gt.size(0)
        loss = (gt - pred).pow(2).view(B, -1).mean()

        return loss, {'loss': loss.detach()}

class MAELoss(Loss):

    def forward(self, pred, gt):
        B   = gt.size(0)
        loss = (gt - pred).abs().view(B, -1).mean()

        return loss, {'loss': loss.detach()}