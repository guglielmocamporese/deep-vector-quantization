# Deep Vector Quantization - What can we do with VQ?

VQ-VAEs use online vector quantization primarily for learning embeddings for image patches. During the learning process, the model learns a quantized set of embeddings that can be used for downstream tasks like image generation (with autoregressive models -> PixelCNNs). In this way, the embeddings learn local features of image patches. 

This project aims at solving these problems:
* What level of learning ability has the VQ framework?
* Can we learn global representations of the entire input with VQ?
* Is it better to use a single VQ feature for classification or a grid of features like in VQVAE?
* Is only the autoencoder capable of learning quantized embeddings or even a classifier?
* Is it possible to learn quantized embeddings with other architectures such as LSTMs or Transformers?


If you use the code of this repo and you find this project useful, please consider to give a star ⭐!

# Results

| Backbone | Quantization | Task          | Decay | Beta | Temp | Dataset | Accuracy |  
| -------- | ------------ | ------------- | ----- | ---- | ---- | ------- | -------- |
| ResNet18 | -            | Classifcation | -     | -    | -    | CIFAR10 | 0.923    |
| ResNet18 | VQ           | Classifcation | -     | 0.25 | -    | CIFAR10 | 0.388    |
| ResNet18 | VQ EMA       | Classifcation | 0.99  | 0.25 | -    | CIFAR10 | 0.909    |
| ResNet18 | Gumbel VQ    | Classifcation | -     | 0.25 | 1.0  | CIFAR10 | 0.879    |

### Reproducibility
All the experiments are **reproducible** since I fixed the initial seed and the learning process is set to be deterministic.
 
  ## Install
```python
# Clone the repo
$ git clone https://github.com/guglielmocamporese/deep-vector-quantization.git deep_vq

# Go to the project directory
$ cd deep_vq
```

### Install dependencies
You need `Python 3.x` , `torch`, `pytorch_lightning` and `torchvision`.  Otherwise, you can install directly with conda all the dependencies with:
```python
# Install the conda env
$ conda env create --file environment.yaml

# Activate the conda env
$ conda activate deep_vq
```

  ## Usage
  ### Train

```python
# No quantization
$ python main.py \
    --mode --train \
    --dataset cifar10
  
# Quantized
$ python main.py \
    --mode --train \
    --dataset cifar10 \
    --vq_mode vq
```

  ### Validate
```python
# No quantization
$ python main.py \
    --mode --validate \
    --dataset cifar10
  
# Quantized
$ python main.py \
    --mode --validate \
    --dataset cifar10 \
    --vq_mode vq
```

`vq_mode` can be:
* not specified, for non quantized network,
* `vq` for standard vector quantization,
* `vq_ema` for vector quantization with exponential moving average,
* `gumbel` for vector quantization with Gumbel trick.

## TO DO:

* [x] Implement quantized networks for classification (inspired by VQVAE paper [[link](https://arxiv.org/abs/1711.00937)]).
  * [x] Implement standard VQ.
  * [x] Implement Exponential Moving Average VQ.
* [x] Implement quantized networks for classification (with Gumbel-Softmax Reparametrization trick [[link](https://arxiv.org/pdf/1611.01144.pdf])]).
  * [x] Implement scheduler for the gumbel temperature.
* [ ] Do ablation on single feat vs grid of features for understanding online vector quantization for deep nets.
* [ ] Do ablation on VQ classifier vs VQ autoencoder (1st stage) + VQ classifier (2nd stage).
