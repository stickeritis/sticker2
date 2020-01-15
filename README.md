# sticker2

**sticker2** is a sequence labeler using
[Transformer](https://arxiv.org/abs/1706.03762) networks. sticker2
models can be trained from scratch or using pretrained models, such as
[BERT](https://arxiv.org/abs/1810.04805v2).

## Introduction

sticker is a sequence labeler that uses either recurrent neural
networks, transformers, or dilated convolution networks. In principle,
it can be used to perform any sequence labeling task, but so far the
focus has been on:

* Part-of-speech tagging
* Morphological tagging
* Topological field tagging
* Lemmatization
* Dependency parsing
* Named entity recognition

## Features

* Input representations: word pieces
* Models representations:
  * Transformers
  * Pretraining (e.g. BERT)
* Multi-task training and classification using scalar weighting.
* Deployment:
  * Standalone binary that links against PyTorch's `libtorch`
  * Very liberal [license](LICENSE.md)

## Status

sticker2 is still **under heavy development**. For production use, we
recommend the [previous version of
sticker](https://github.com/stickeritis/sticker).

## References

sticker uses techniques from or was inspired by the following papers:

* [Viable Dependency Parsing as Sequence
  Labeling](https://www.aclweb.org/anthology/papers/N/N19/N19-1077/). Michalina
  Strzyz, David Vilares, Carlos Gómez-Rodríguez, 2019, Proceedings of
  the 2019 Conference of the North American Chapter of the Association
  for Computational Linguistics: Human Language Technologies
* More to be added...

## Issues

You can report bugs and feature requests in the [sticker2 issue
tracker](https://github.com/stickeritis/sticker2/issues).

## License

sticker is licensed under the [Blue Oak Model License version
1.0.0](LICENSE.md). The Tensorflow protocol buffer definitions in
`tf-proto` are licensed under the Apache License version 2.0. The
[list of contributors](CONTRIBUTORS) is also available.
