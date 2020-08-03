use std::borrow::Borrow;
use std::collections::HashMap;
#[cfg(feature = "load-hdf5")]
use std::path;

#[cfg(feature = "load-hdf5")]
use anyhow::Error;
#[cfg(feature = "load-hdf5")]
use hdf5::File;
#[cfg(feature = "load-hdf5")]
use sticker_transformers::hdf5_model::LoadFromHDF5;
use sticker_transformers::layers::Dropout;
use sticker_transformers::models::bert::{
    BertConfig, BertEmbeddings, BertEncoder, BertError, BertLayerOutput,
};
use sticker_transformers::models::roberta::RobertaEmbeddings;
use sticker_transformers::models::sinusoidal::SinusoidalEmbeddings;
use sticker_transformers::models::Encoder;
use tch::nn::{ModuleT, Path};
use tch::{self, Tensor};

use crate::config::{PositionEmbeddings, PretrainConfig};
use crate::encoders::Encoders;
use crate::model::seq_classifiers::{SequenceClassifiers, SequenceClassifiersLoss};

pub trait PretrainBertConfig {
    fn bert_config(&self) -> &BertConfig;
}

impl PretrainBertConfig for PretrainConfig {
    fn bert_config(&self) -> &BertConfig {
        match self {
            PretrainConfig::Bert(config) => config,
            PretrainConfig::XlmRoberta(config) => config,
        }
    }
}

#[derive(Debug)]
enum BertEmbeddingLayer {
    Bert(BertEmbeddings),
    Roberta(RobertaEmbeddings),
    Sinusoidal(SinusoidalEmbeddings),
}

impl BertEmbeddingLayer {
    fn new<'a>(
        vs: impl Borrow<Path<'a>>,
        pretrain_config: &PretrainConfig,
        position_embeddings: PositionEmbeddings,
    ) -> Self {
        match (pretrain_config, position_embeddings) {
            (PretrainConfig::Bert(config), PositionEmbeddings::Model) => {
                BertEmbeddingLayer::Bert(BertEmbeddings::new(vs, config))
            }
            (PretrainConfig::Bert(config), PositionEmbeddings::Sinusoidal { normalize }) => {
                let normalize = if normalize { Some(2.) } else { None };
                BertEmbeddingLayer::Sinusoidal(SinusoidalEmbeddings::new(vs, config, normalize))
            }
            (PretrainConfig::XlmRoberta(config), PositionEmbeddings::Model) => {
                BertEmbeddingLayer::Roberta(RobertaEmbeddings::new(vs, config))
            }
            (PretrainConfig::XlmRoberta(_), PositionEmbeddings::Sinusoidal { .. }) => {
                unreachable!()
            }
        }
    }

    #[cfg(feature = "load-hdf5")]
    fn load_from_hdf5<'a>(
        vs: impl Borrow<Path<'a>>,
        pretrain_config: &PretrainConfig,
        pretrained_file: &File,
    ) -> Result<BertEmbeddingLayer, Error> {
        let vs = vs.borrow();

        let embeddings = match pretrain_config {
            PretrainConfig::Bert(config) => {
                BertEmbeddingLayer::Bert(BertEmbeddings::load_from_hdf5(
                    vs.sub("encoder"),
                    config,
                    pretrained_file.group("bert/embeddings")?,
                )?)
            }
            PretrainConfig::XlmRoberta(config) => {
                BertEmbeddingLayer::Roberta(RobertaEmbeddings::load_from_hdf5(
                    vs.sub("encoder"),
                    config,
                    pretrained_file.group("bert/embeddings")?,
                )?)
            }
        };

        Ok(embeddings)
    }
}

impl ModuleT for BertEmbeddingLayer {
    fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
        use BertEmbeddingLayer::*;

        match self {
            Bert(ref embeddings) => embeddings.forward_t(input, train),
            Roberta(ref embeddings) => embeddings.forward_t(input, train),
            Sinusoidal(ref embeddings) => embeddings.forward_t(input, train),
        }
    }
}

/// Multi-task classifier using the BERT architecture with scalar weighting.
#[derive(Debug)]
pub struct BertModel {
    embeddings: BertEmbeddingLayer,
    encoder: BertEncoder,
    seq_classifiers: SequenceClassifiers,
    layers_dropout: Dropout,
}

impl BertModel {
    /// Construct a fresh model.
    ///
    /// `layer_dropout` is the probability with which layers should
    /// be dropped out in scalar weighting during training.
    pub fn new<'a>(
        vs: impl Borrow<Path<'a>>,
        pretrain_config: &PretrainConfig,
        encoders: &Encoders,
        layers_dropout: f64,
        position_embeddings: PositionEmbeddings,
    ) -> Result<Self, BertError> {
        let vs = vs.borrow();

        let bert_config = pretrain_config.bert_config();

        let embeddings =
            BertEmbeddingLayer::new(vs.sub("encoder"), pretrain_config, position_embeddings);
        let encoder = BertEncoder::new(vs.sub("encoder"), bert_config)?;
        let seq_classifiers = SequenceClassifiers::new(vs, pretrain_config, encoders);

        Ok(BertModel {
            embeddings,
            encoder,
            layers_dropout: Dropout::new(layers_dropout),
            seq_classifiers,
        })
    }

    #[cfg(feature = "load-hdf5")]
    /// Construct a model and load parameters from a pretrained model.
    ///
    /// `layer_dropout` is the probability with which layers should
    /// be dropped out in scalar weighting during training.
    pub fn from_pretrained<'a>(
        vs: impl Borrow<Path<'a>>,
        pretrain_config: &PretrainConfig,
        hdf_path: impl AsRef<path::Path>,
        encoders: &Encoders,
        layers_dropout: f64,
    ) -> Result<Self, Error> {
        let vs = vs.borrow();

        let pretrained_file = File::open(hdf_path)?;

        let bert_config = pretrain_config.bert_config();

        let embeddings = BertEmbeddingLayer::load_from_hdf5(vs, pretrain_config, &pretrained_file)?;

        let encoder = BertEncoder::load_from_hdf5(
            vs.sub("encoder"),
            bert_config,
            pretrained_file.group("bert/encoder")?,
        )?;

        let seq_classifiers = SequenceClassifiers::new(vs, pretrain_config, encoders);

        Ok(BertModel {
            embeddings,
            encoder,
            layers_dropout: Dropout::new(layers_dropout),
            seq_classifiers,
        })
    }

    /// Encode an input.
    fn encode(
        &self,
        inputs: &Tensor,
        attention_mask: &Tensor,
        train: bool,
        freeze_layers: FreezeLayers,
    ) -> Vec<BertLayerOutput> {
        let embeds = if freeze_layers.embeddings {
            tch::no_grad(|| self.embeddings.forward_t(inputs, train))
        } else {
            self.embeddings.forward_t(inputs, train)
        };

        let mut encoded = if freeze_layers.encoder {
            tch::no_grad(|| self.encoder.encode(&embeds, Some(&attention_mask), train))
        } else {
            self.encoder.encode(&embeds, Some(&attention_mask), train)
        };

        for layer in &mut encoded {
            layer.output = if freeze_layers.classifiers {
                tch::no_grad(|| self.layers_dropout.forward_t(&layer.output, train))
            } else {
                self.layers_dropout.forward_t(&layer.output, train)
            };
        }

        encoded
    }

    /// Compute the logits for a batch of inputs.
    ///
    /// * `attention_mask`: specifies which sequence elements should
    ///    be masked when applying the encoder.
    /// * `train`: indicates whether this forward pass will be used
    ///   for backpropagation.
    /// * `freeze_embeddings`: exclude embeddings from backpropagation.
    /// * `freeze_encoder`: exclude the encoder from backpropagation.
    pub fn logits(
        &self,
        inputs: &Tensor,
        attention_mask: &Tensor,
        train: bool,
        freeze_layers: FreezeLayers,
    ) -> HashMap<String, Tensor> {
        let encoding = self.encode(inputs, attention_mask, train, freeze_layers);
        self.seq_classifiers.forward_t(&encoding, train)
    }

    /// Compute the loss given a batch of inputs and target labels.
    ///
    /// * `attention_mask`: specifies which sequence elements should
    ///    be masked when applying the encoder.
    /// * `token_mask`: specifies which sequence elements should be
    ///    masked when computing the loss. Typically, this is used
    ///    to exclude padding and continuation word pieces.
    /// * `targets`: the labels to be predicted, per encoder name.
    /// * `label_smoothing`: apply label smoothing, redistributing
    ///   the given probability to non-target labels.
    /// * `train`: indicates whether this forward pass will be used
    ///   for backpropagation.
    /// * `freeze_embeddings`: exclude embeddings from backpropagation.
    /// * `freeze_encoder`: exclude the encoder from backpropagation.
    #[allow(clippy::too_many_arguments)]
    pub fn loss(
        &self,
        inputs: &Tensor,
        attention_mask: &Tensor,
        token_mask: &Tensor,
        targets: &HashMap<String, Tensor>,
        label_smoothing: Option<f64>,
        train: bool,
        freeze_layers: FreezeLayers,
        include_continuations: bool,
    ) -> SequenceClassifiersLoss {
        let encoding = self.encode(inputs, attention_mask, train, freeze_layers);

        if freeze_layers.classifiers {
            tch::no_grad(|| {
                self.seq_classifiers.loss(
                    &encoding,
                    attention_mask,
                    token_mask,
                    targets,
                    label_smoothing,
                    train,
                    include_continuations,
                )
            })
        } else {
            self.seq_classifiers.loss(
                &encoding,
                attention_mask,
                token_mask,
                targets,
                label_smoothing,
                train,
                include_continuations,
            )
        }
    }

    /// Compute the top-k labels for each encoder for the input.
    ///
    /// * `attention_mask`: specifies which sequence elements should
    ///    be masked when applying the encoder.
    pub fn top_k(
        &self,
        inputs: &Tensor,
        attention_mask: &Tensor,
    ) -> HashMap<String, (Tensor, Tensor)> {
        let encoding = self.encode(
            inputs,
            attention_mask,
            false,
            FreezeLayers {
                embeddings: true,
                encoder: true,
                classifiers: true,
            },
        );

        self.seq_classifiers.top_k(&encoding)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FreezeLayers {
    pub embeddings: bool,
    pub encoder: bool,
    pub classifiers: bool,
}
