
from .differential import SurfaceMapLoss
from .mixin import Loss

class IsometricSurfaceMapLoss(Loss, SurfaceMapLoss):

    def __init__(self, **kwargs):
        Loss.__init__(self, **kwargs)
        SurfaceMapLoss.__init__(self)

    def map_distortion(self, FFF):
        return self.symmetric_dirichlet(FFF)


class ARAPSurfaceMapLoss(IsometricSurfaceMapLoss):

    def map_distortion(self, FFF):
        return self.arap(FFF)


class ConformalSurfaceMapLoss(IsometricSurfaceMapLoss):

    def map_distortion(self, FFF):
        return self.conformal(FFF)


class EquiarealSurfaceMapLoss(IsometricSurfaceMapLoss):

    def map_distortion(self, FFF):
        return self.equiareal(FFF)
