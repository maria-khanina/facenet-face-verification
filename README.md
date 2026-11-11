# FaceNet from scratch: face verification

FaceNet (NN2 variant) implemented in PyTorch from its building blocks and trained with a custom training loop for face verification: given two face images, decide whether they show the same person.

## What is inside

- **Architecture:** convolution blocks with batch normalization, L2 pooling, and inception modules that produce L2-normalized 128-dimensional embeddings.
- **Training loop:** fp16 mixed precision with `torch.amp` autocast and `GradScaler`, triplet loss with hard-pair mining and an M-per-class sampler (`pytorch-metric-learning`), linear learning-rate warmup, and restoration of the best validation weights.
- **Baseline:** an ImageNet-pretrained ResNet-18 retrained for face embeddings with the same loop.
- **Evaluation:** ROC AUC and best-threshold accuracy on pairs of images, with the final test on 6,000 pairs from Labeled Faces in the Wild.

## Results

The models were trained on about 17,000 images of 105 people.

| Model | LFW ROC AUC | Best-threshold accuracy |
|---|---|---|
| FaceNet NN2, trained from scratch | 0.724 | 0.672 |
| ResNet-18, pretrained on ImageNet | 0.835 | 0.760 |

With this small training set, the pretrained backbone generalizes better. Stronger weight decay in Adam did not stabilize the validation score of the model trained from scratch.


## Package and scripts

The notebook code is split into the `face_verification` package. All numbers live in `training_settings.py` and `architecture_settings.py`.

| Command | What it does |
|---|---|
| `python -m scripts.split_identities` | splits the 105 training identities into 90 for training and 15 for validation (`splits/pins_identity_split.json`), so validation pairs show people the model has not seen |
| `python -m scripts.train_model --model nn2 --pool-backend triton --compile --run-name <name>` | trains and evaluates on LFW; writes `runs/<name>/history.json` and `summary.json` |
| `python -m scripts.profile_training_step --pool-backend eager [--compile]` | times data loading against GPU work over 30 steps, then ranks operators by GPU time with `torch.profiler` (`runs/profiles/<name>/`) |
| `python -m scripts.benchmark_l2_pool` | times L2 pooling alone (forward and backward) at the input shapes of the network (`runs/l2_pool_benchmark.json`) |

Validation and LFW pairs are drawn with fixed seeds, so every run is scored on the same pairs. The best-threshold accuracy counts a pair whose distance equals the threshold as the same person, as `roc_curve` does; the notebook used a strict comparison.

`slurm/` runs the experiments as Slurm jobs on a single-GPU node inside WSL2: `setup_environment.sh` creates the environment and copies the data to the Linux file system, and `submit_experiments.sh` queues the tests, the benchmark, four profiles and two training runs.

## Training speed: profiling, torch.compile and a fused Triton kernel

Measured on an 8 GB laptop GPU (24 streaming multiprocessors, compute capability 8.9) with batch 128, fp16 autocast and 160×160 images.

### Where the time goes

`torch.profiler` on the eager step (139 ms on the GPU) shows that data loading is not the bottleneck: the loader waits about 5 ms per step. Convolutions take the largest share (forward and backward together about 52 ms). The rest are memory-bound kernels:

- conversions between the NCHW and NHWC layouts around cuDNN convolutions;
- max pooling backward;
- batch normalization;
- the L2 pooling chain, which runs in float32: autocast keeps `pow` in float32, so the window sum and its backward read and write float32 tensors.

### A fused L2 pooling kernel

L2 pooling with a 3×3 window, stride 1 and padding 1 computes y = sqrt(sum of x² over the window + ε). In eager mode this is three kernels forward and three backward, and autograd keeps the squared input for the backward pass. `face_verification/l2_pool_triton.py` replaces them with one Triton kernel each way:

- **Forward:** loads the window, sums the squares in float32 registers and writes only y in the input dtype.
- **Backward:** uses ∂y_w/∂x_i = x_i / y_w. The input gradient is x_i times the sum of grad_y / y over the 3×3 neighbours of i. Every program writes its own elements, so no atomic additions are needed, and only x and y are kept for backward.

Tests compare the output and the input gradient with the eager version in float32 and float16.

L2 pooling alone, forward and backward, batch 128:

| Input shape (layers) | Eager | torch.compile | Triton | Extra memory: eager / compile / Triton |
|---|---|---|---|---|
| 256×20×20 (1) | 8.85 ms | 2.41 ms | **1.21 ms** | 325 / 200 / 25 MiB |
| 640×10×10 (4) | 4.90 ms | 2.37 ms | **0.69 ms** | 203 / 125 / 16 MiB |
| 1024×5×5 (1) | 1.37 ms | 0.75 ms | **0.26 ms** | 81 / 51 / 6 MiB |

Part of the gain over eager comes from writing float16 instead of float32.

### The whole training step

| Variant | GPU time per step | Peak GPU memory |
|---|---|---|
| eager | 138.8 ms | 2.16 GiB |
| Triton L2 pooling | 114.2 ms (−18%) | 1.60 GiB |
| torch.compile | 102.4 ms (−26%) | 1.38 GiB |
| torch.compile with Triton L2 pooling | 103.5 ms (−25%) | **1.02 GiB** |

What the numbers show:

- **Without compilation**, the kernel alone removes 18% of the step. That is more than the time of the L2 pooling kernels themselves, because the float32 copies around the chain also disappear.
- **Under torch.compile** the kernel does not make the step faster. Inductor already fuses the eager chain, and the remaining time is in cuDNN convolutions, which Inductor calls as external kernels and cannot fuse with their neighbours. The kernel still cuts peak memory by 26% compared with compilation alone.
- **A pitfall under compilation.** A first version of the kernel ran five times slower inside the compiled graph (1.65 ms instead of 0.33 ms for one forward call). Inductor passed the float `epsilon` argument as fp64, so the sum and the square root ran in double precision, which is slow on consumer GPUs; eager Triton launches pass Python floats as fp32. ε is now a compile-time constant (`tl.constexpr`).
- **Still open:** Inductor also fills the output buffer of the user kernel with NaN before every call, an extra full write per layer.

### Full training runs

Twenty epochs each:

| Variant | Seconds per epoch | Images per second | Peak GPU memory | Best validation ROC AUC | LFW ROC AUC |
|---|---|---|---|---|---|
| eager | 17.3 | 867 | 2.19 GiB | 0.837 | 0.735 |
| torch.compile with Triton L2 pooling | 13.1 | 1140 | 1.08 GiB | 0.835 | 0.701 |

The first compiled epoch takes 23 s because of compilation. Training order is not seeded, so one run per variant cannot separate an effect of the variant on accuracy from run-to-run variation.

Under WSL2, pinned host memory for the data loader fails with `cuMemHostAlloc` out of memory, so `PIN_LOADER_MEMORY` is off; all numbers above are measured with it off.

## Tests, style and types

```
pytest -q
pycodestyle --max-line-length=88 face_verification scripts tests tools
python tools/rules_check.py .
pyright
```
