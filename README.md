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

