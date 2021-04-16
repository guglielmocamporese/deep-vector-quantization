# Deep Vector Quantization

  ## Install
```python
# Clone the repo
$ git clone https://github.com/guglielmocamporese/deep-vector-quantization.git deep_vq

# Go to the project directory
$ cd deep_vq
```

Install dependencies

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
