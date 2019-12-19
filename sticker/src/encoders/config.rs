use std::collections::HashMap;
use std::ops::Deref;

use serde::{Deserialize, Serialize};
use sticker_encoders::layer::Layer;
use sticker_encoders::lemma::BackoffStrategy;

/// Configuration of a set of encoders.
///
/// The configuration is a mapping from encoder name to
/// encoder configuration.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncodersConfig(pub HashMap<String, EncoderType>);

impl Deref for EncodersConfig {
    type Target = HashMap<String, EncoderType>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// The type of encoder.
#[serde(rename_all = "lowercase")]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum EncoderType {
    /// Encoder for syntactical dependencies.
    Dependency(DependencyEncoder),

    /// Lemma encoder using edit trees.
    Lemma(BackoffStrategy),

    /// Encoder for plain sequence labels.
    Sequence(Layer),
}

/// The type of dependency encoder.
#[serde(rename_all = "lowercase")]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum DependencyEncoder {
    /// Encode a token's head by relative position.
    RelativePosition,

    /// Encode a token's head by relative position of the POS tag.
    RelativePOS,
}
