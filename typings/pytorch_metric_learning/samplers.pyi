from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler

class MPerClassSampler(Sampler[int]):
    def __init__(
        self,
        labels: Sequence[int],
        m: int,
        batch_size: int,
        length_before_new_iter: int,
    ) -> None: ...
    def __iter__(self) -> Iterator[int]: ...
    def __len__(self) -> int: ...
