import numpy
from numpy.typing import NDArray

def roc_curve(
    y_true: NDArray[numpy.int64],
    y_score: NDArray[numpy.float32],
) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64], NDArray[numpy.float64]]: ...
def auc(x: NDArray[numpy.float64], y: NDArray[numpy.float64]) -> float: ...
