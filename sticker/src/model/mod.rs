use std::borrow::Borrow;
use std::collections::HashMap;
use std::path;

use failure::Fallible;
use hdf5::File;
use sticker_transformers::bert_model::{
    BertConfig, BertEmbeddings, BertEncoder, BertError, BertLayerOutput,
};
use sticker_transformers::hdf5_model::LoadFromHDF5;
use sticker_transformers::layers::Dropout;
use sticker_transformers::scalar_weighting::ScalarWeightClassifier;
use tch::nn::{ModuleT, Path};
use tch::{self, Kind, Tensor};

use crate::encoders::Encoders;

/// Multi-task classifier using the BERT architecture with scalar weighting.
#[derive(Debug)]
pub struct BertModel {
    embeddings: BertEmbeddings,
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
        config: &BertConfig,
        encoders: &Encoders,
        layers_dropout: f64,
    ) -> Result<Self, BertError> {
        let vs = vs.borrow();

        let embeddings = BertEmbeddings::new(vs.sub("encoder"), config);
        let encoder = BertEncoder::new(vs.sub("encoder"), config)?;

        let classifiers = encoders
            .iter()
            .map(|encoder| {
                (
                    encoder.name().to_owned(),
                    ScalarWeightClassifier::new(
                        vs.sub("classifiers")
                            .sub(format!("{}_classifier", encoder.name())),
                        config.num_hidden_layers,
                        config.hidden_size,
                        encoder.encoder().len() as i64,
                        0.1,
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
        config: &BertConfig,
        hdf_path: impl AsRef<path::Path>,
        encoders: &Encoders,
        layers_dropout: f64,
    ) -> Fallible<Self> {
        let vs = vs.borrow();

        let pretrained_file = File::open(hdf_path, "r")?;

        let embeddings = BertEmbeddings::load_from_hdf5(
            vs.sub("encoder"),
            config,
            pretrained_file.group("bert/embeddings")?,
        )?;

        let encoder = BertEncoder::load_from_hdf5(
            vs.sub("encoder"),
            config,
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
                        config.num_hidden_layers,
                        config.hidden_size,
                        encoder.encoder().len() as i64,
                        0.1,
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
            tch::no_grad(|| self.embeddings.forward_t(inputs, None, None, train))
        } else {
            self.embeddings.forward_t(inputs, None, None, train)
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
            let loss = (loss * &token_mask).sum(Kind::Float) / &token_mask_sum;
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
