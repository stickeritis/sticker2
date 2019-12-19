use std::collections::HashMap;
use std::ops::Deref;

use conllx::graph::Sentence;
use edit_tree::EditTree;
use failure::Fallible;
use numberer::Numberer;
use serde::{Deserialize, Serialize};
use sticker_encoders::categorical::MutableCategoricalEncoder;
use sticker_encoders::deprel::{
    DependencyEncoding, RelativePOS, RelativePOSEncoder, RelativePosition, RelativePositionEncoder,
};
use sticker_encoders::layer::LayerEncoder;
use sticker_encoders::lemma::EditTreeEncoder;
use sticker_encoders::SentenceEncoder;

use crate::encoders::{DependencyEncoder, EncoderType, EncodersConfig};

/// Wrapper of the various supported encoders.
#[derive(Deserialize, Serialize)]
pub enum Encoder {
    Lemma(MutableCategoricalEncoder<EditTreeEncoder, EditTree<char>>),
    Layer(MutableCategoricalEncoder<LayerEncoder, String>),
    RelativePOS(MutableCategoricalEncoder<RelativePOSEncoder, DependencyEncoding<RelativePOS>>),
    RelativePosition(
        MutableCategoricalEncoder<RelativePositionEncoder, DependencyEncoding<RelativePosition>>,
    ),
}

impl SentenceEncoder for Encoder {
    type Encoding = usize;

    fn encode(&self, sentence: &Sentence) -> Fallible<Vec<Self::Encoding>> {
        match self {
            Encoder::Layer(encoder) => encoder.encode(sentence),
            Encoder::Lemma(encoder) => encoder.encode(sentence),
            Encoder::RelativePOS(encoder) => encoder.encode(sentence),
            Encoder::RelativePosition(encoder) => encoder.encode(sentence),
        }
    }
}

impl From<&EncoderType> for Encoder {
    fn from(encoder_type: &EncoderType) -> Self {
        // We start labeling at 2. 0 is reserved for padding, 1 for continuations.
        match encoder_type {
            EncoderType::Dependency(DependencyEncoder::RelativePOS) => Encoder::RelativePOS(
                MutableCategoricalEncoder::new(RelativePOSEncoder, Numberer::new(2)),
            ),
            EncoderType::Dependency(DependencyEncoder::RelativePosition) => {
                Encoder::RelativePosition(MutableCategoricalEncoder::new(
                    RelativePositionEncoder,
                    Numberer::new(2),
                ))
            }
            EncoderType::Lemma(backoff_strategy) => Encoder::Lemma(MutableCategoricalEncoder::new(
                EditTreeEncoder::new(*backoff_strategy),
                Numberer::new(2),
            )),
            EncoderType::Sequence(ref layer) => Encoder::Layer(MutableCategoricalEncoder::new(
                LayerEncoder::new(layer.clone()),
                Numberer::new(2),
            )),
        }
    }
}

/// A set of encoders
///
/// This set is a mapping from encoder names to encoders.
#[derive(Serialize, Deserialize)]
pub struct Encoders(HashMap<String, Encoder>);

impl From<&EncodersConfig> for Encoders {
    fn from(config: &EncodersConfig) -> Self {
        Encoders(
            config
                .iter()
                .map(|(name, encoder_type)| (name.clone(), encoder_type.into()))
                .collect(),
        )
    }
}

impl Deref for Encoders {
    type Target = HashMap<String, Encoder>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
