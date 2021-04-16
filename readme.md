# Deep Vector Quantization - Can we use Deep Vector Quantization for Classification?

VQ-VAEs use online vector quantization primarly for learning embeddings for image patches. During the learning process the model learns a quantized set of embeddings that can be used for down stream tasks like image generation (with autoregressive models -> PixelCNNs). In this way the embeddings learn local features of image patches. 
Can we learn a global representation of the entire image with VQ?
 
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
    --quantized
```

  ### Validate
```python
# No quantization
python main.py \
    --mode --validate \
    --dataset cifar10
  
# Quantized
python main.py \
    --mode --validate \
    --dataset cifar10 \
    --quantized
```

## TO DO:

* [x] Implement quantized networks for classification (inspired by VQVAE paper [[link](https://arxiv.org/abs/1711.00937)])

* [ ] Implement quantized networks for classification (with Gumbel-Softmax Reparametrization trick [[link](https://arxiv.org/pdf/1611.01144.pdf])])

* [ ] Get full understanding of online vector quantization for deep nets.

* [ ] Get results of quantized networks on standard iamge classification benchmarks:
  * [x] MNIST
  * [x] CIFAR10
  * [x] CIFAR100
  * ImageNet
