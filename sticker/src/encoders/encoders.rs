use std::hash::Hash;
use std::ops::Deref;

use conllx::graph::Sentence;
use edit_tree::EditTree;
use failure::Fallible;
use numberer::Numberer;
use serde::{Deserialize, Serialize};
use sticker_encoders::categorical::{ImmutableCategoricalEncoder, MutableCategoricalEncoder};
use sticker_encoders::deprel::{
    DependencyEncoding, RelativePOS, RelativePOSEncoder, RelativePosition, RelativePositionEncoder,
};
use sticker_encoders::layer::LayerEncoder;
use sticker_encoders::lemma::EditTreeEncoder;
use sticker_encoders::{EncodingProb, SentenceDecoder, SentenceEncoder};

use crate::encoders::{DependencyEncoder, EncoderType, EncodersConfig};

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
pub enum CategoricalEncoderWrap<E, V>
where
    V: Clone + Eq + Hash,
{
    Immutable(ImmutableCategoricalEncoder<E, V>),
    Mutable(MutableCategoricalEncoder<E, V>),
}

impl<E, V> From<MutableCategoricalEncoder<E, V>> for CategoricalEncoderWrap<E, V>
where
    V: Clone + Eq + Hash,
{
    fn from(encoder: MutableCategoricalEncoder<E, V>) -> Self {
        CategoricalEncoderWrap::Mutable(encoder)
    }
}

impl<D> SentenceDecoder for CategoricalEncoderWrap<D, D::Encoding>
where
    D: SentenceDecoder,
    D::Encoding: Clone + Eq + Hash,
{
    type Encoding = usize;

    fn decode<S>(&self, labels: &[S], sentence: &mut Sentence) -> Fallible<()>
    where
        S: AsRef<[EncodingProb<Self::Encoding>]>,
    {
        match self {
            CategoricalEncoderWrap::Immutable(decoder) => decoder.decode(labels, sentence),
            CategoricalEncoderWrap::Mutable(decoder) => decoder.decode(labels, sentence),
        }
    }
}

impl<E> SentenceEncoder for CategoricalEncoderWrap<E, E::Encoding>
where
    E: SentenceEncoder,
    E::Encoding: Clone + Eq + Hash,
{
    type Encoding = usize;

    fn encode(&self, sentence: &Sentence) -> Fallible<Vec<Self::Encoding>> {
        match self {
            CategoricalEncoderWrap::Immutable(encoder) => encoder.encode(sentence),
            CategoricalEncoderWrap::Mutable(encoder) => encoder.encode(sentence),
        }
    }
}

impl<E, V> CategoricalEncoderWrap<E, V>
where
    V: Clone + Eq + Hash,
{
    pub fn len(&self) -> usize {
        match self {
            CategoricalEncoderWrap::Immutable(encoder) => encoder.len(),
            CategoricalEncoderWrap::Mutable(encoder) => encoder.len(),
        }
    }
}

/// Wrapper of the various supported encoders.
#[derive(Deserialize, Serialize)]
pub enum Encoder {
    Lemma(CategoricalEncoderWrap<EditTreeEncoder, EditTree<char>>),
    Layer(CategoricalEncoderWrap<LayerEncoder, String>),
    RelativePOS(CategoricalEncoderWrap<RelativePOSEncoder, DependencyEncoding<RelativePOS>>),
    RelativePosition(
        CategoricalEncoderWrap<RelativePositionEncoder, DependencyEncoding<RelativePosition>>,
    ),
}

#[allow(clippy::len_without_is_empty)]
impl Encoder {
    pub fn len(&self) -> usize {
        match self {
            Encoder::Layer(encoder) => encoder.len(),
            Encoder::Lemma(encoder) => encoder.len(),
            Encoder::RelativePOS(encoder) => encoder.len(),
            Encoder::RelativePosition(encoder) => encoder.len(),
        }
    }
}

impl SentenceDecoder for Encoder {
    type Encoding = usize;

    fn decode<S>(&self, labels: &[S], sentence: &mut Sentence) -> Fallible<()>
    where
        S: AsRef<[EncodingProb<Self::Encoding>]>,
    {
        match self {
            Encoder::Layer(decoder) => decoder.decode(labels, sentence),
            Encoder::Lemma(decoder) => decoder.decode(labels, sentence),
            Encoder::RelativePOS(decoder) => decoder.decode(labels, sentence),
            Encoder::RelativePosition(decoder) => decoder.decode(labels, sentence),
        }
    }
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
                MutableCategoricalEncoder::new(RelativePOSEncoder, Numberer::new(2)).into(),
            ),
            EncoderType::Dependency(DependencyEncoder::RelativePosition) => {
                Encoder::RelativePosition(
                    MutableCategoricalEncoder::new(RelativePositionEncoder, Numberer::new(2))
                        .into(),
                )
            }
            EncoderType::Lemma(backoff_strategy) => Encoder::Lemma(
                MutableCategoricalEncoder::new(
                    EditTreeEncoder::new(*backoff_strategy),
                    Numberer::new(2),
                )
                .into(),
            ),
            EncoderType::Sequence(ref layer) => Encoder::Layer(
                MutableCategoricalEncoder::new(LayerEncoder::new(layer.clone()), Numberer::new(2))
                    .into(),
            ),
        }
    }
}

/// A named encoder.
#[derive(Deserialize, Serialize)]
pub struct NamedEncoder {
    encoder: Encoder,
    name: String,
}

impl NamedEncoder {
    /// Get the encoder.
    pub fn encoder(&self) -> &Encoder {
        &self.encoder
    }

    /// Get the encoder name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// A collection of named encoders.
#[derive(Serialize, Deserialize)]
pub struct Encoders(Vec<NamedEncoder>);

impl From<&EncodersConfig> for Encoders {
    fn from(config: &EncodersConfig) -> Self {
        Encoders(
            config
                .iter()
                .map(|encoder| NamedEncoder {
                    name: encoder.name.clone(),
                    encoder: (&encoder.encoder).into(),
                })
                .collect(),
        )
    }
}

impl Deref for Encoders {
    type Target = [NamedEncoder];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
