
from .binary import BinaryCheckpointing
from .html import CollectionVisualizationCheckpointing
from .image import ImageCheckpointing
from .kaolin import KaolinCheckpointing
from .rendering import BlenderCheckpointing
from .report import ReportCheckpointing
from .surface import SurfaceCheckpointing


class AllCheckpointing(BinaryCheckpointing, CollectionVisualizationCheckpointing,
                        ImageCheckpointing, SurfaceCheckpointing,
                        ReportCheckpointing, BlenderCheckpointing,
                        KaolinCheckpointing):
    pass