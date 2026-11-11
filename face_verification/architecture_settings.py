"""Layer sizes of the FaceNet NN2 network (Schroff et al., 2015, table 2).

An inception module is described by its input channels, the 1x1 branch, the
3x3 reduction and 3x3 branch, the 5x5 reduction and 5x5 branch (built as a 3x3
convolution, as in the paper's implementation), the pool projection and the
pool type. A reduction module has no 1x1 branch and no pool projection and
halves the spatial size.
"""
from typing import Literal

PoolType = Literal["max_pool", "l2_pool"]
InceptionLayout = tuple[int, int, int, int, int, int, int, PoolType]
ReductionLayout = tuple[int, int, int, int, int]

IMAGE_CHANNELS = 3
EMBEDDING_SIZE = 128
RESNET_EMBEDDING_INPUT = 512

STEM_CONVOLUTION_CHANNELS = 64
STEM_CONVOLUTION_KERNEL = 7
STEM_CONVOLUTION_STRIDE = 2
STEM_CONVOLUTION_PADDING = 3
STEM_POOL_KERNEL = 3
STEM_POOL_STRIDE = 2
SECOND_STAGE_CHANNELS = 192
SMALL_KERNEL = 3
SMALL_PADDING = 1
REDUCTION_STRIDE = 2

INCEPTION_STAGE_THREE: tuple[InceptionLayout, ...] = (
    (192, 64, 96, 128, 16, 32, 32, "max_pool"),
    (256, 64, 96, 128, 32, 64, 64, "l2_pool"),
)
REDUCTION_STAGE_THREE: ReductionLayout = (320, 128, 256, 32, 64)
INCEPTION_STAGE_FOUR: tuple[InceptionLayout, ...] = (
    (640, 256, 96, 192, 32, 64, 128, "l2_pool"),
    (640, 224, 112, 224, 32, 64, 128, "l2_pool"),
    (640, 192, 128, 256, 32, 64, 128, "l2_pool"),
    (640, 160, 144, 288, 32, 64, 128, "l2_pool"),
)
REDUCTION_STAGE_FOUR: ReductionLayout = (640, 160, 256, 64, 128)
INCEPTION_STAGE_FIVE: tuple[InceptionLayout, ...] = (
    (1024, 384, 192, 384, 48, 128, 128, "l2_pool"),
    (1024, 384, 192, 384, 48, 128, 128, "max_pool"),
)
FINAL_CHANNELS = 1024
