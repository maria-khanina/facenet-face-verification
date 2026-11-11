from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler

class MPerClassSampler(Sampler[list[int]]):
    def __init__(
        self,
        labels: Sequence[int],
        m: int,
        batch_size: int,
        num_batches: int,
    ) -> None: ...
    def __iter__(self) -> Iterator[list[int]]: ...
    def __len__(self) -> int: ...
