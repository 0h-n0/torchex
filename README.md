[![GitHub license](https://img.shields.io/github/license/0h-n0/torchex.svg)](https://github.com/0h-n0/torchex)
[![PYTHON version](https://img.shields.io/badge/python-3.5,3.6-blue.svg)](https://github.com/0h-n0/torchex)
[![PyPI version](https://img.shields.io/pypi/v/torchex.svg)](https://badge.fury.io/py/torchex)
[![CircleCI](https://circleci.com/gh/0h-n0/torchex.svg?style=svg&circle-token=99e93ba7bf6433d0cd33adbec2fbd042d141353d)](https://circleci.com/gh/0h-n0/torchex)
[![Build Status](https://travis-ci.org/0h-n0/torchex.svg?branch=master)](https://travis-ci.org/0h-n0/torchex)
[![codecov](https://codecov.io/gh/0h-n0/torchex/branch/master/graph/badge.svg)](https://codecov.io/gh/0h-n0/torchex)
[![Documentation Status](https://readthedocs.org/projects/torchex/badge/?version=latest)](https://torchex.readthedocs.io/en/latest/?badge=latest)
[![Maintainability](https://api.codeclimate.com/v1/badges/7cd6c99f10d22db13ee8/maintainability)](https://codeclimate.com/github/0h-n0/torchex/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/7cd6c99f10d22db13ee8/test_coverage)](https://codeclimate.com/github/0h-n0/torchex/test_coverage)
[![BCH compliance](https://bettercodehub.com/edge/badge/0h-n0/torchex?branch=master)](https://bettercodehub.com/)
[![Downloads](https://img.shields.io/pypi/dm/torchex.svg)](https://pypi.org/project/torchex/)

# (WIP) Pytorch Extension library

Pytorch Extenstion library provides advanced Neural Network Layers. You can easily use them like using original pytorch.

## Installation

```
$ git clone --recursive 
$ cd torchex
$ pip install -e .
```


## Requirements

* Pytorch >= 0.4.1

## Documentation

After executing the following codes, you can read `PytorchExtenstion` documentation in docs/build/html/index.html. 

```shell
$ pip install -r requirements.txt
$ cd docs
$ make html
```

## TODO

### Layers

- [x] support fundamental complex operations
  - to_complex method
  - to_real method
  - complex_norm method
- [ ] add submodule for many examples.
- [ ] SeparableConv2D
- [ ] LocallyConnected1D
- [x] Highway
- [x] Inception
- [x] InceptionBN
- [x] Conv2dLocal
- [x] MLPConv2d
  * [Network In Network](https://arxiv.org/abs/1312.4400v3)
- [ ] NaryTreeLSTM
- [ ] StatefulZoneoutLSTM
- [ ] StatefulPeepholeLSTM
- [ ] StatefulMGU
- [ ] BinaryHierarchicalSoftmax
- [ ] BlackOut
- [ ] CRF1d
- [ ] SimplifiedDropconnect
- [ ] Swish
- [ ] NegativeSampling
- [ ] ResidualCell
- [ ] Attention Cell
  * [XiaoIce Band: A Melody and Arrangement Generation Framework for Pop Music](https://www.kdd.org/kdd2018/accepted-papers/view/xiaoice-banda-melody-and-arrangement-generation-framework-for-pop-music)
- [ ] MLP Cell
  * same as above.
- [ ] DFT2d
  * [Rotation Equivariance and Invariance in Convolutional Neural Networks](https://arxiv.org/pdf/1805.12301.pdf)
  * https://github.com/bchidest/RiCNN/tree/master/ricnn
- [ ] My original DFT layer (made by Koji Ono)
  - [ ] DFT1d
  - [x] DFT2d
  - [ ] DFT3d  
  - [ ] iDFT1d
  - [ ] iDFT2d
  - [ ] iDFT3d  
  - [ ] RFFT1d
  - [ ] RFFT2d
  - [ ] RFFT3d  

- [ ] Conic Convolutional Layers
  * same as above.

### Zoo

- [x] ImageTransferNet
  
### Optimizer

- [ ] chainer.optimizer_hooks.GradientLARS

### Atiributions

- [x] Integrated Gradients

## Examples


## Related Projects

* torchhp
  * Hyper-Parameter Turning Library for Pytorch.
* torchrl
  * Pytorch Reinforcement Learning Library.
* torchchem
  * TorchChem aims to provide a high quality open-source toolchain that democratizes the use of deep-learning in drug discovery, materials science, quantum chemistry, and biology.
* torchml
  * Auto model optimization library for pytorch.
* torcdata
  * Pytorch Datasets.

## Codes References 

* Chainer
  * One of the most wonderfull DeepLearning framework.
  * https://github.com/chainer/chainer
* NLP
  * allenNLP
    * https://github.com/allenai/allennlp
  * fairseq
    * https://github.com/pytorch/fairseq
  * text
    * https://github.com/pytorch/text
  * translate
    * https://github.com/pytorch/translate
* Audio
  * neural_sp
    * https://github.com/hirofumi0810/neural_sp/
  * deepspeech.pytorch
    * https://github.com/SeanNaren/deepspeech.pytorch
  * Awesome Speech Recognition Speech Synthesis Papers
    * https://github.com/zzw922cn/awesome-speech-recognition-speech-synthesis-papers
  * speech
    * https://github.com/awni/speech
  * pytorch-asr
    * https://github.com/jinserk/pytorch-asr