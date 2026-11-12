"""Group the GPU kernels of every saved profile by the part of the network.

Reads runs/profiles/<run>/operators.txt, keeps the rows of GPU kernels and
memory copies (operator rows such as aten::convolution would count the same
time twice), merges the duplicate rows that the profiler writes for Triton
kernels (the same name, or the name with the _0 suffix of the copy that
torch.compile makes of a user kernel), and sums their time per step into
groups by kernel name. Writes runs/profiles/kernel_groups.json.

Usage (from the repository root):
  python -m scripts.summarize_profiles
"""
import json
from pathlib import Path
import re

from face_verification.training_settings import RUNS_DIRECTORY

ROW_PATTERN = re.compile(r"\s*(?P<milliseconds>[\d.]+) ms\s+\d+\s+(?P<name>.*)")
COPY_SUFFIX = re.compile(r"(_kernel)_\d+$")
SKIPPED_PREFIXES = ("aten::", "Optimizer.", "ProfilerStep", "cuda", "Torch-Compiled")
GROUP_MARKERS: list[tuple[str, tuple[str, ...]]] = [
    ("host-to-GPU batch copy", ("memcpy htod",)),
    ("L2 pooling", ("l2_pool", "avg_pool2d")),
    ("batch norm with fused neighbours", ("batch_norm", "bn_fw", "bn_bw")),
    ("max pooling", ("max_pool",)),
    ("NCHW/NHWC conversions", ("nchwtonhwc", "nhwctonchw")),
    ("convolutions (cuDNN and cuBLAS)", ("xmma", "gemm", "cudnn", "cutlass")),
    ("optimizer", ("adam", "multi_tensor", "foreach")),
]


def read_kernel_rows(path: Path) -> dict[str, float]:
    """Return the time per step of every kernel row, one entry per name."""
    kernels: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = ROW_PATTERN.match(line)
        if match is None or match["name"].startswith(SKIPPED_PREFIXES):
            continue
        name = COPY_SUFFIX.sub(r"\1", match["name"].strip())
        kernels.setdefault(name, float(match["milliseconds"]))
    return kernels


def group_of(name: str) -> str:
    """Return the group of a kernel by the words in its name."""
    lowered = name.lower()
    for group, markers in GROUP_MARKERS:
        if any(marker in lowered for marker in markers):
            return group
    return "other"


def group_kernels(kernels: dict[str, float]) -> dict[str, float]:
    """Sum the kernel time per group, rounded to hundredths of a millisecond."""
    groups: dict[str, float] = {}
    for name, milliseconds in kernels.items():
        group = group_of(name)
        groups[group] = groups.get(group, 0.0) + milliseconds
    return {group: round(total, 2) for group, total in groups.items()}


def main() -> None:
    """Group every saved profile and write the table."""
    directory = RUNS_DIRECTORY / "profiles"
    table: dict[str, dict[str, float]] = {
        path.parent.name: group_kernels(read_kernel_rows(path))
        for path in sorted(directory.glob("*/operators.txt"))
    }
    text = json.dumps(table, indent=2) + "\n"
    (directory / "kernel_groups.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
