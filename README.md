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
| `python -m scripts.train_model --model nn2 --pool-backend eager --run-name <name>` | trains and evaluates on LFW; writes `runs/<name>/history.json` and `summary.json` |
| `python -m scripts.profile_training_step --pool-backend eager --run-name <name>` | times data loading against GPU work over 30 steps, then ranks kernels by GPU time with `torch.profiler` (`runs/profiles/<name>/`) |
| `python -m scripts.summarize_profiles` | groups the kernel time of every saved profile (`runs/profiles/kernel_groups.json`) |
| `python -m scripts.benchmark_l2_pool` | times L2 pooling alone in fp16 autocast and in float32 (`runs/l2_pool_benchmark.json`) |

Training and profiling take switches for each speed optimization: `--compile`, `--pool-backend triton`, `--transfer-uint8`, `--cudnn-benchmark`, `--compile-optimizer`, `--merge-pointwise`.

Validation and LFW pairs are drawn with fixed seeds, so every run is scored on the same pairs. The best-threshold accuracy counts a pair whose distance equals the threshold as the same person, as `roc_curve` does; the notebook used a strict comparison.

`slurm/` runs the experiments as Slurm jobs on a single-GPU node inside WSL2: `setup_environment.sh` creates the environment and copies the data to the Linux file system, and `submit_experiments.sh` queues the tests, the kernel benchmark, the profiles and two training runs.

## Training speed

Profiling the eager step, then the step under torch.compile, points to what is left to optimize; each optimization is measured on top of torch.compile and then all together (8 GB laptop GPU, batch 128, fp16 autocast):

| Variant | GPU time per step | Peak memory |
|---|---|---|
| eager | 141.6 ms | 2.16 GiB |
| torch.compile | 104.9 ms | 1.39 GiB |
| torch.compile + uint8 batch transfer, cuDNN benchmark, compiled optimizer step, merged 1×1 convolutions, fused Triton L2 pooling | **92.0 ms** | **1.04 GiB** |

- torch.compile fuses element-wise work and batch norm with its neighbours, removes the concatenation copies and keeps activations channels-last, so the layout conversions around cuDNN disappear.
- The four optimizations aimed at the remaining costs save 2–6 ms each; the fused L2 pooling kernels (`face_verification/l2_pool_triton.py`, registered with `torch.library.triton_op` and layout-independent) remove 19% of the eager step, and under torch.compile cut peak memory by a quarter at the same speed.
- Full training: 11.2 s per epoch against 17.4 s in eager, with the same LFW score.

Profiles, the analysis of the compiled graph, the remaining costs and the kernel design are in [docs/training_performance.md](docs/training_performance.md).

## Tests, style and types

```
pytest -q
pycodestyle --max-line-length=88 face_verification scripts tests tools
python tools/rules_check.py .
pyright
```
