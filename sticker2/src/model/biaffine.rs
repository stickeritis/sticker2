use std::borrow::Borrow;

use serde::{Deserialize, Serialize};
use sticker_transformers::models::BertConfig;
use sticker_transformers::scalar_weighting::ScalarWeight;
use sticker_transformers::traits::LayerOutput;
use tch::nn::{Init, Linear, Module, Path};
use tch::{Kind, Tensor};

use crate::config::PretrainConfig;
use crate::model::bert::PretrainBertConfig;
use crate::model::pairwise_bilinear::{PairwiseBilinear, PairwiseBilinearConfig};

/// Biaffine layer
///
/// This layer computes twice a batch of matrices *[batch_size,
/// max_seq_len, max_seq_len]* containing pairwire dependency arc and
/// label scores.
pub struct Biaffine {
    scalar_weight: ScalarWeight,

    arc_dependent: Linear,
    arc_head: Linear,
    label_dependent: Linear,
    label_head: Linear,

    bilinear_arc: PairwiseBilinear,
    bilinear_label: PairwiseBilinear,
}

impl Biaffine {
    pub fn new<'a>(
        vs: impl Borrow<Path<'a>>,
        pretrain_config: &PretrainConfig,
        biaffine_config: &BiaffineConfig,
    ) -> Self {
        let bert_config = pretrain_config.bert_config();

        let vs = vs.borrow();

        let scalar_weight = ScalarWeight::new(
            vs / "scalar_weight",
            bert_config.num_hidden_layers,
            bert_config.hidden_dropout_prob,
        );

        let arc_dependent = Self::affine(
            vs / "arc_dependent",
            bert_config,
            bert_config.hidden_size,
            biaffine_config.arc_dims as i64,
            "weight",
            "bias",
        );

        let arc_head = Self::affine(
            vs / "arc_head",
            bert_config,
            bert_config.hidden_size,
            biaffine_config.arc_dims as i64,
            "weight",
            "bias",
        );

        let label_dependent = Self::affine(
            vs / "label_dependent",
            bert_config,
            bert_config.hidden_size,
            biaffine_config.label_dims as i64,
            "weight",
            "bias",
        );

        let label_head = Self::affine(
            vs / "label_head",
            bert_config,
            bert_config.hidden_size,
            biaffine_config.label_dims as i64,
            "weight",
            "bias",
        );

        let bilinear_arc = PairwiseBilinear::new(
            vs / "pairwise_bilinear_arc",
            &PairwiseBilinearConfig {
                initializer_range: bert_config.initializer_range,
                in_features: biaffine_config.arc_dims as i64,
                out_features: 1,
            },
        );

        let bilinear_label = PairwiseBilinear::new(
            vs / "pairwise_bilinear_label",
            &PairwiseBilinearConfig {
                initializer_range: bert_config.initializer_range,
                in_features: biaffine_config.label_dims as i64,
                out_features: biaffine_config.n_labels as i64,
            },
        );

        Biaffine {
            scalar_weight,

            arc_dependent,
            arc_head,
            label_dependent,
            label_head,

            bilinear_arc,
            bilinear_label,
        }
    }

    fn affine<'a>(
        vs: impl Borrow<Path<'a>>,
        config: &BertConfig,
        in_features: i64,
        out_features: i64,
        weight_name: &str,
        bias_name: &str,
    ) -> Linear {
        let vs = vs.borrow();

        Linear {
            ws: vs.var(
                weight_name,
                &[out_features, in_features],
                Init::Randn {
                    mean: 0.,
                    stdev: config.initializer_range,
                },
            ),
            bs: vs.var(bias_name, &[out_features], Init::Const(0.)),
        }
    }

    pub fn forward(
        &self,
        layers: &[impl LayerOutput],
        token_mask: &Tensor,
        train: bool,
    ) -> BiaffineScores {
        let layer0 = layers[0].layer_output();
        let seq_len = layer0.size()[1];

        let token_mask = logits_mask(&token_mask);

        let self_attachment_mask =
            Tensor::eye(seq_len, (Kind::Float, layer0.device())).unsqueeze(0) * -10_000.;

        // Get weighted hidden representation.
        let hidden = self.scalar_weight.forward(layers, train);

        // Compute dependent/head arc representations of each token.
        let arc_dependent = self.arc_dependent.forward(&hidden);
        let arc_head = self.arc_head.forward(&hidden);

        // From from these representations, compute the arc score matrix.
        let mut arc_score_logits = self
            .bilinear_arc
            .forward(&arc_head, &arc_dependent)
            .squeeze1(-1);

        // Mask out self-attachments.
        arc_score_logits += self_attachment_mask;

        // Mask out non-initial word pieces.
        arc_score_logits += token_mask.unsqueeze(2) + token_mask.unsqueeze(1);

        // Compute dependent/head label representations of each token.
        let label_dependent = self.label_dependent.forward(&hidden);
        let label_head = self.label_head.forward(&hidden);

        // From from these representations, compute the label score matrix.
        let label_score_logits = self.bilinear_label.forward(&label_head, &label_dependent);

        BiaffineScores {
            arc: arc_score_logits,
            label: label_score_logits,
        }
    }

    /// Compute the biaffine classifier loss.
    ///
    /// * `gold_arcs` has the shape *[batch_size, max_seq_len]* and
    ///   stores for each sequence element its head. The `ROOT` token
    ///   must be omitted.
    pub fn loss(
        &self,
        layers: &[impl LayerOutput],
        gold_arcs: &Tensor,
        gold_labels: &Tensor,
        token_mask: &Tensor,
        train: bool,
    ) -> Tensor {
        // Todo: we are not computing accuracies below yet...
        // Todo: add a representation for root.
        // Todo: add label dimension

        let scores = self.forward(layers, token_mask, train);

        let shape = scores.arc.size();
        let batch_size = shape[0];
        let seq_len = shape[1];

        let log_arc_scores = scores
            .arc
            .view([batch_size, -1])
            .log_softmax(-1, Kind::Float)
            .view([batch_size, seq_len, seq_len]);

        let arc_loss = -(gold_arcs * &log_arc_scores);

        let log_label_scores = scores
            .label
            .view([batch_size, -1])
            .log_softmax(-1, Kind::Float)
            .view([batch_size, seq_len, seq_len]);

        let label_loss = -(gold_labels * &log_label_scores);

        arc_loss + label_loss
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BiaffineConfig {
    pub arc_dims: usize,
    pub label_dims: usize,
    pub n_labels: usize,
}

#[derive(Debug)]
pub struct BiaffineScores {
    pub arc: Tensor,
    pub label: Tensor,
}

fn logits_mask(mask: &Tensor) -> Tensor {
    (1.0 - mask.to_kind(Kind::Float)) * -10_000.
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::iter;

    use maplit::btreeset;
    use sticker_transformers::models::BertConfig;
    use sticker_transformers::traits::LayerOutput;
    use tch::nn::VarStore;
    use tch::{Device, Kind, Tensor};

    use super::{Biaffine, BiaffineConfig};
    use crate::config::PretrainConfig;

    pub struct TestLayerOutput(Tensor);

    impl LayerOutput for TestLayerOutput {
        fn layer_output(&self) -> &Tensor {
            &self.0
        }
    }

    fn biaffine_variables() -> BTreeSet<String> {
        btreeset![
            "arc_dependent.bias".to_string(),
            "arc_dependent.weight".to_string(),
            "arc_head.bias".to_string(),
            "arc_head.weight".to_string(),
            "label_dependent.bias".to_string(),
            "label_dependent.weight".to_string(),
            "label_head.bias".to_string(),
            "label_head.weight".to_string(),
            "pairwise_bilinear_arc.weight".to_string(),
            "pairwise_bilinear_label.weight".to_string(),
            "scalar_weight.layer_weights".to_string(),
            "scalar_weight.scale".to_string()
        ]
    }

    fn bert_config() -> BertConfig {
        BertConfig {
            attention_probs_dropout_prob: 0.1,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            hidden_size: 64,
            initializer_range: 0.02,
            intermediate_size: 3072,
            layer_norm_eps: 1e-12,
            max_position_embeddings: 512,
            num_attention_heads: 2,
            num_hidden_layers: 4,
            type_vocab_size: 2,
            vocab_size: 30000,
        }
    }

    fn varstore_variables(vs: &VarStore) -> BTreeSet<String> {
        vs.variables()
            .into_iter()
            .map(|(k, _)| k)
            .collect::<BTreeSet<_>>()
    }

    #[test]
    fn biaffine_layer_creates_expected_variable_names() {
        let vs = VarStore::new(Device::Cpu);

        let pretrain_config = PretrainConfig::Bert(bert_config());

        let _ = Biaffine::new(
            vs.root(),
            &pretrain_config,
            &BiaffineConfig {
                arc_dims: 768,
                label_dims: 256,
                n_labels: 5,
            },
        );

        assert_eq!(varstore_variables(&vs), biaffine_variables());
    }

    #[test]
    fn biaffine_layer_shapes_are_correct() {
        let vs = VarStore::new(Device::Cpu);

        let pretrain_config_bert = bert_config();
        let pretrain_config = PretrainConfig::Bert(bert_config());

        let biaffine = Biaffine::new(
            vs.root(),
            &pretrain_config,
            &BiaffineConfig {
                arc_dims: 64,
                label_dims: 32,
                n_labels: 5,
            },
        );

        let layers = iter::repeat_with(|| {
            TestLayerOutput(Tensor::rand(
                &[4, 10, pretrain_config_bert.hidden_size],
                (Kind::Float, Device::Cpu),
            ))
        })
        .take(pretrain_config_bert.num_hidden_layers as usize)
        .collect::<Vec<_>>();

        let scores = biaffine.forward(
            &layers,
            &Tensor::randint(2, &[4, 10], (Kind::Int64, Device::Cpu)),
            false,
        );

        assert_eq!(scores.arc.size(), &[4, 10, 10]);
        assert_eq!(scores.label.size(), &[4, 10, 10, 5]);
    }
}
