from torch import nn, Tensor

class TripletMarginLoss(nn.Module):
    def __init__(self, margin: float) -> None: ...
    def __call__(
        self,
        embeddings: Tensor,
        labels: Tensor,
        indices_tuple: tuple[Tensor, ...],
    ) -> Tensor: ...
