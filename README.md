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


* The model architecture and training regime was largely based on [75
  Languages, 1 Model: Parsing Universal Dependencies
  Universally](https://www.aclweb.org/anthology/D19-1279.pdf).  Dan
  Kondratyuk and Milan Straka, 2019, Proceedings of the EMNLP 2019 and
  the 9th IJCNLP.
* The tagging as sequence labeling scheme was proposed by [Dependency
  Parsing as a Sequence Labeling
  Task](https://www.degruyter.com/downloadpdf/j/pralin.2010.94.issue--1/v10108-010-0017-3/v10108-010-0017-3.pdf). Drahomíra
  Spoustová, Miroslav Spousta, 2010, The Prague Bulletin of
  Mathematical Linguistics, Volume 94.
* The idea to combine this scheme with neural networks comes from
  [Viable Dependency Parsing as Sequence
  Labeling](https://www.aclweb.org/anthology/papers/N/N19/N19-1077/). Michalina
  Strzyz, David Vilares, Carlos Gómez-Rodríguez, 2019, Proceedings of
  the 2019 Conference of the North American Chapter of the Association
  for Computational Linguistics: Human Language Technologies
* The encoding of lemmatization as edit trees was proposed in [Towards
  a Machine-Learning Architecture for Lexical Functional Grammar
  Parsing](http://grzegorz.chrupala.me/papers/phd-single.pdf).
  Grzegorz Chrupała, 2008, PhD dissertation, Dublin City University.


## Issues

You can report bugs and feature requests in the [sticker2 issue
tracker](https://github.com/stickeritis/sticker2/issues).

## License

sticker is licensed under the [Blue Oak Model License version
1.0.0](LICENSE.md). The Tensorflow protocol buffer definitions in
`tf-proto` are licensed under the Apache License version 2.0. The
[list of contributors](CONTRIBUTORS) is also available.
