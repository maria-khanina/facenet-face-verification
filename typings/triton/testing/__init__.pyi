from collections.abc import Callable
from typing import Any

def do_bench(
    function: Callable[[], Any],
    warmup: int,
    rep: int,
) -> float: ...
