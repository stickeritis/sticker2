//! CoNLL-X layer encoder.

use conllx::graph::{Node, Sentence};
use conllx::token::Features;
use conllx::token::Token;
use failure::Error;
use serde::{Deserialize, Serialize};

use super::{EncodingProb, SentenceDecoder, SentenceEncoder};

mod error;
use self::error::*;

/// Tagging layer.
#[serde(rename_all = "lowercase")]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum Layer {
    CPos,
    Pos,
    Feature(String),
}

/// Layer values.
pub trait LayerValue {
    /// Set a layer value.
    fn set_value(&mut self, layer: &Layer, value: impl Into<String>);

    /// Get a layer value.
    fn value(&self, layer: &Layer) -> Option<&str>;
}

impl LayerValue for Token {
    /// Set the layer to the given value.
    fn set_value(&mut self, layer: &Layer, value: impl Into<String>) {
        let value = value.into();

        match layer {
            Layer::CPos => {
                self.set_cpos(Some(value));
            }
            Layer::Pos => {
                self.set_pos(Some(value));
            }
            Layer::Feature(ref feature) => {
                if self.features().is_none() {
                    self.set_features(Some(Features::default()));
                }

                self.features_mut()
                    .unwrap()
                    .insert(feature.clone(), Some(value));
            }
        };
    }

    /// Look up the layer value in a token.
    fn value(&self, layer: &Layer) -> Option<&str> {
        match layer {
            Layer::CPos => self.cpos(),
            Layer::Pos => self.pos(),
            Layer::Feature(ref feature) => {
                self.features()?.get(feature)?.as_ref().map(String::as_str)
            }
        }
    }
}

/// Encode sentences using a CoNLL-X layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayerEncoder {
    layer: Layer,
}

impl LayerEncoder {
    /// Construct a new layer encoder of the given layer.
    pub fn new(layer: Layer) -> Self {
        LayerEncoder { layer }
    }
}

impl SentenceDecoder for LayerEncoder {
    type Encoding = String;

    fn decode<S>(&self, labels: &[S], sentence: &mut Sentence) -> Result<(), Error>
    where
        S: AsRef<[EncodingProb<Self::Encoding>]>,
    {
        assert_eq!(
            labels.len(),
            sentence.len() - 1,
            "Labels and sentence length mismatch"
        );

        for (token, token_labels) in sentence
            .iter_mut()
            .filter_map(Node::token_mut)
            .zip(labels.iter())
        {
            if let Some(label) = token_labels.as_ref().get(0) {
                token.set_value(&self.layer, label.encoding().as_str());
            }
        }

        Ok(())
    }
}

impl SentenceEncoder for LayerEncoder {
    type Encoding = String;

    fn encode(&self, sentence: &Sentence) -> Result<Vec<Self::Encoding>, Error> {
        let mut encoding = Vec::with_capacity(sentence.len() - 1);
        for token in sentence.iter().filter_map(Node::token) {
            let label = token
                .value(&self.layer)
                .ok_or_else(|| EncodeError::MissingLabel {
                    form: token.form().to_owned(),
                })?;
            encoding.push(label.to_owned());
        }

        Ok(encoding)
    }
}
