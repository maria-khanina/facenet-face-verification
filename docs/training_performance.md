# Training-step performance of FaceNet NN2

Measured on an 8 GB laptop GPU (24 streaming multiprocessors, compute capability 8.9) under WSL2, batch 128, fp16 autocast, 160×160 images. Every number comes from `runs/profiles/<run>/summary.json` and `operators.txt` (`scripts/profile_training_step.py`), the kernel groups from `runs/profiles/kernel_groups.json` (`scripts/summarize_profiles.py`), the isolated kernel timings from `runs/l2_pool_benchmark.json` (`scripts/benchmark_l2_pool.py`). The data loader runs without pinned host memory: pinned allocation fails under WSL2.

## 1. Eager

GPU time per step 141.6 ms, peak memory 2.16 GiB. The loader is not the bottleneck (about 5 ms of waiting per step).

| Kernel group | ms per step |
|---|---|
| Convolutions (cuDNN) | 35.2 |
| Other element-wise work (ReLU, additions, concatenation, casts) | 37.4 |
| Max pooling | 12.5 |
| NCHW/NHWC conversions around cuDNN | 12.3 |
| Host-to-GPU copy of the batch | 11.1 |
| L2 pooling | 10.8 |
| Batch norm | 10.7 |
| Optimizer | 2.3 |

## 2. torch.compile

GPU time per step 104.9 ms (−26%), peak memory 1.39 GiB. From the generated code (`TORCH_LOGS=output_code`):

- **Fusion.** Element-wise operations and reductions become about 350 generated Triton kernels. Batch norm keeps a kernel boundary between its statistics (two-stage reductions) and the normalization, which is fused with the following ReLU. Each branch of an inception module writes its batch norm and ReLU output straight into its slice of the concatenated tensor, so `torch.cat` copies nothing. L2 pooling becomes one kernel each way, with the square computed inside the preceding batch-norm kernel.
- **Layout.** Activations are kept channels-last (NHWC), so the NCHW/NHWC conversions disappear. That, not faster convolutions, is most of the drop in convolution time.
- **Convolutions** stay calls to cuDNN (external kernels), which nothing can be fused into.
- **Saved tensors.** The backward pass recomputes cheap element-wise values (for example the square and the root of L2 pooling) instead of keeping them.
- **What is left out.** `optimizer.step()` runs outside the compiled model.

What remains after compilation:

| Kernel group | ms per step |
|---|---|
| Convolutions (cuDNN) | 36.3 |
| Batch norm with fused neighbours | 25.5 |
| Max pooling | 10.9 |
| Host-to-GPU copy of the batch | 9.9 |
| L2 pooling | 5.1 |
| Optimizer | 2.4 |
| Other | 2.8 |

## 3. Optimizations derived from the compiled profile

| Remaining cost | Optimization | Expected effect |
|---|---|---|
| Host-to-GPU copy, 9.9 ms: the batch is 39 MB of float32 | send uint8 pixels and normalize on the GPU; pixels are uint8 after decoding and resizing, so nothing is lost | about four times fewer bytes |
| Convolutions, 36 ms, algorithms chosen by cuDNN heuristics | `torch.backends.cudnn.benchmark`: time every algorithm once per input shape and keep the fastest | a few ms |
| Optimizer outside the compiled graph, 2.4 ms | compile the optimizer step (learning rate in a tensor, so the warmup does not recompile it) | most of 2.4 ms |
| Three 1×1 convolutions over the same input in every inception module | one merged 1×1 convolution, split after batch norm and ReLU (both per channel, so the result is the same) | the input read once, one wider matrix multiplication |
| L2 pooling, 5.1 ms | the fused Triton kernels (section 4) | about 1 ms, and memory |

Max pooling (10.9 ms) is the largest group without a change here. About 6 ms of it is the max-pooling backward of the two stem poolings, which Inductor implements as a zero fill of the gradient buffer followed by a scatter; a fused backward that writes every input gradient once is the next candidate.

## 4. Fused L2 pooling kernels

L2 pooling with a 3×3 window, stride 1 and padding 1 computes y = sqrt(Σ x² + ε) over every window. `face_verification/l2_pool_triton.py` implements one Triton kernel for each pass:

- **Forward** loads the nine neighbours, sums the squares in float32 registers and writes only y.
- **Backward** uses ∂y_w/∂x_i = x_i / y_w: the input gradient is x_i times the sum of grad_y / y over the nine windows containing i. Every program writes only its own elements, so there are no atomic additions; only x and y are kept for backward.
- **Layout.** Neighbour addresses come from the tensor strides, so the same kernels run on NCHW (eager) and NHWC (compiled) tensors without conversions.
- **Registration.** Both passes are `torch.library.triton_op` operators with registered autograd, so torch.compile sees their inputs, outputs and layouts. ε is a compile-time constant: passed as an argument, Inductor gives a Python float to the kernel as fp64.

Tests compare outputs and gradients with eager pooling in float32, float16, channels-last and inside torch.compile.

L2 pooling alone (forward and backward, six layers of the network summed):

| Precision | Eager | torch.compile of this function | Triton |
|---|---|---|---|
| fp16 tensors, fp16 autocast | 29.4 ms | 12.9 ms | 4.3 ms |
| float32 tensors | 25.1 ms | 15.0 ms | 5.3 ms |

The kernels are faster in float32 as well, so the gain comes from fusion rather than precision. Under autocast the eager chain runs in float32 (autocast keeps `pow` in float32), which widens its gap. Compiling this function alone understates torch.compile: inside the whole network Inductor also fuses the square into the preceding kernel, and there L2 pooling takes 5.1 ms per step against 4.1 ms with the Triton kernels.

## 5. Results

| Variant | GPU time per step | Peak memory |
|---|---|---|
| eager | 141.6 ms | 2.16 GiB |
| eager + Triton L2 pooling | 114.7 ms | 1.61 GiB |
| torch.compile | 104.9 ms | 1.39 GiB |
| + uint8 batch | 98.5 ms | 1.38 GiB |
| + cuDNN benchmark | 100.1 ms | 1.38 GiB |
| + compiled optimizer | 102.8 ms | 1.38 GiB |
| + merged 1×1 convolutions | 101.9 ms | 1.42 GiB |
| + Triton L2 pooling | 104.9 ms | 1.02 GiB |
| torch.compile + all of the above | **92.0 ms** | **1.04 GiB** |

Each "+" row adds one optimization to torch.compile alone. Without compilation the Triton kernels remove 19% of the step; under compilation they do not change the step time but cut peak memory by a quarter. The other four optimizations each save 2–6 ms, and together with the kernels the step is 12% faster than torch.compile alone and 35% faster than eager.

Full training, 20 epochs (`runs/nn2_eager`, `runs/nn2_all_optimizations`):

| Variant | Seconds per epoch | Images per second | First epoch | Peak memory | LFW ROC AUC |
|---|---|---|---|---|---|
| eager | 17.4 | 859 | 18.1 s | 2.19 GiB | 0.680 |
| all optimizations | 11.2 | 1336 | 26.7 s | 2.26 GiB | 0.684 |

The first epoch includes compilation and the cuDNN algorithm search. Peak memory in full training includes the evaluation passes with batch 256, which are not part of the profiled step. Training order is not seeded: two eager runs gave LFW ROC AUC 0.735 and 0.680, so single runs do not separate an effect of a variant on accuracy from run-to-run variation.
