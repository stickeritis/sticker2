use std::borrow::Borrow;
use std::collections::HashMap;
use std::path;

use failure::Fallible;
use hdf5::File;
use sticker_transformers::hdf5_model::LoadFromHDF5;
use sticker_transformers::layers::Dropout;
use sticker_transformers::models::bert::{
    BertConfig, BertEmbeddings, BertEncoder, BertError, BertLayerOutput,
};
use sticker_transformers::models::roberta::RobertaEmbeddings;
use sticker_transformers::models::sinusoidal::SinusoidalEmbeddings;
use sticker_transformers::scalar_weighting::{
    ScalarWeightClassifier, ScalarWeightClassifierConfig,
};
use tch::nn::{ModuleT, Path};
use tch::{self, Kind, Tensor};

use crate::config::{PositionEmbeddings, PretrainConfig};
use crate::encoders::Encoders;

trait PretrainBertConfig {
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
                BertEmbeddingLayer::Bert(BertEmbeddings::new(vs, config, true))
            }
            (PretrainConfig::Bert(config), PositionEmbeddings::Sinusoidal) => {
                BertEmbeddingLayer::Sinusoidal(SinusoidalEmbeddings::new(vs, config))
            }
            (PretrainConfig::XlmRoberta(config), PositionEmbeddings::Model) => {
                BertEmbeddingLayer::Roberta(RobertaEmbeddings::new(vs, config))
            }
            (PretrainConfig::XlmRoberta(_), PositionEmbeddings::Sinusoidal) => unreachable!(),
        }
    }

    fn load_from_hdf5<'a>(
        vs: impl Borrow<Path<'a>>,
        pretrain_config: &PretrainConfig,
        pretrained_file: &File,
    ) -> Fallible<BertEmbeddingLayer> {
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
    classifiers: HashMap<String, ScalarWeightClassifier>,
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

        let classifiers = encoders
            .iter()
            .map(|encoder| {
                (
                    encoder.name().to_owned(),
                    ScalarWeightClassifier::new(
                        vs.sub("classifiers")
                            .sub(format!("{}_classifier", encoder.name())),
                        &ScalarWeightClassifierConfig {
                            dropout_prob: bert_config.hidden_dropout_prob,
                            hidden_size: bert_config.hidden_size,
                            input_size: bert_config.hidden_size,
                            layer_dropout_prob: 0.1,
                            layer_norm_eps: bert_config.layer_norm_eps,
                            n_layers: bert_config.num_hidden_layers,
                            n_labels: encoder.encoder().len() as i64,
                        },
                    ),
                )
            })
            .collect();

        Ok(BertModel {
            classifiers,
            embeddings,
            encoder,
            layers_dropout: Dropout::new(layers_dropout),
        })
    }

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
    ) -> Fallible<Self> {
        let vs = vs.borrow();

        let pretrained_file = File::open(hdf_path, "r")?;

        let bert_config = pretrain_config.bert_config();

        let embeddings = BertEmbeddingLayer::load_from_hdf5(vs, pretrain_config, &pretrained_file)?;

        let encoder = BertEncoder::load_from_hdf5(
            vs.sub("encoder"),
            bert_config,
            pretrained_file.group("bert/encoder")?,
        )?;

        let classifiers = encoders
            .iter()
            .map(|encoder| {
                (
                    encoder.name().to_owned(),
                    ScalarWeightClassifier::new(
                        vs.sub("classifiers")
                            .sub(format!("{}_classifier", encoder.name())),
                        &ScalarWeightClassifierConfig {
                            dropout_prob: bert_config.hidden_dropout_prob,
                            hidden_size: bert_config.hidden_size,
                            input_size: bert_config.hidden_size,
                            layer_dropout_prob: 0.1,
                            layer_norm_eps: bert_config.layer_norm_eps,
                            n_layers: bert_config.num_hidden_layers,
                            n_labels: encoder.encoder().len() as i64,
                        },
                    ),
                )
            })
            .collect();

        Ok(BertModel {
            embeddings,
            encoder,
            layers_dropout: Dropout::new(layers_dropout),
            classifiers,
        })
    }

    /// Encode an input.
    fn encode(
        &self,
        inputs: &Tensor,
        attention_mask: &Tensor,
        train: bool,
        freeze_embeddings: bool,
        freeze_encoder: bool,
    ) -> Vec<BertLayerOutput> {
        let embeds = if freeze_embeddings {
            tch::no_grad(|| self.embeddings.forward_t(inputs, train))
        } else {
            self.embeddings.forward_t(inputs, train)
        };

        let mut encoded = if freeze_encoder {
            tch::no_grad(|| {
                self.encoder
                    .forward_t(&embeds, Some(&attention_mask), train)
            })
        } else {
            self.encoder
                .forward_t(&embeds, Some(&attention_mask), train)
        };

        for layer in &mut encoded {
            layer.output = self.layers_dropout.forward_t(&layer.output, train);
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
        freeze_embeddings: bool,
        freeze_encoder: bool,
    ) -> HashMap<String, Tensor> {
        let encoding = self.encode(
            inputs,
            attention_mask,
            train,
            freeze_embeddings,
            freeze_encoder,
        );

        self.classifiers
            .iter()
            .map(|(encoder_name, classifier)| {
                (
                    encoder_name.to_string(),
                    classifier.logits(&encoding, train),
                )
            })
            .collect()
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
        freeze_embeddings: bool,
        freeze_encoder: bool,
        include_continuations: bool,
    ) -> (Tensor, HashMap<String, Tensor>, HashMap<String, Tensor>) {
        let encoding = self.encode(
            inputs,
            attention_mask,
            train,
            freeze_embeddings,
            freeze_encoder,
        );

        let token_mask = token_mask.to_kind(Kind::Float);
        let token_mask_sum = token_mask.sum(Kind::Float);

        let mut encoder_specific_loss = HashMap::with_capacity(self.classifiers.len());
        let mut encoder_specific_accuracy = HashMap::with_capacity(self.classifiers.len());
        for (encoder_name, classifier) in &self.classifiers {
            let (loss, correct) =
                classifier.losses(&encoding, &targets[encoder_name], label_smoothing, train);
            let loss = if include_continuations {
                (loss * attention_mask).sum(Kind::Float) / &attention_mask.sum(Kind::Float)
            } else {
                (loss * &token_mask).sum(Kind::Float) / &token_mask_sum
            };
            let acc = (correct * &token_mask).sum(Kind::Float) / &token_mask_sum;

            encoder_specific_loss.insert(encoder_name.clone(), loss);
            encoder_specific_accuracy.insert(encoder_name.clone(), acc);
        }

        let summed_loss = encoder_specific_loss.values().fold(
            Tensor::zeros(&[], (Kind::Float, inputs.device())),
            |summed_loss, loss| summed_loss + loss,
        );

        (
            summed_loss,
            encoder_specific_loss,
            encoder_specific_accuracy,
        )
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
        let encoding = self.encode(inputs, attention_mask, false, true, true);
        self.classifiers
            .iter()
            .map(|(encoder_name, classifier)| {
                let (probs, mut labels) = classifier
                    .forward(&encoding, false)
                    // Exclude first two classes (padding and continuation).
                    .slice(-1, 2, -1, 1)
                    .topk(3, -1, true, true);

                // Fix label offsets.
                labels += 2;

                (
                    encoder_name.to_string(),
                    // XXX: make k configurable
                    (probs, labels),
                )
            })
            .collect()
    }
}
