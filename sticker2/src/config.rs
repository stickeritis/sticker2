use std::convert::TryFrom;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use sentencepiece::SentencePieceProcessor;
use serde::{Deserialize, Serialize};
use sticker_transformers::models::albert::AlbertConfig;
use sticker_transformers::models::bert::BertConfig;
use wordpieces::WordPieces;

use crate::encoders::EncodersConfig;
use crate::error::StickerError;
use crate::input::{AlbertTokenizer, BertTokenizer, Tokenize, XlmRobertaTokenizer};

/// Input configuration.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    /// The type of tokenizer to use.
    pub tokenizer: Tokenizer,
}

/// Labeler configuration.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Labeler {
    /// The encoder labels file.
    pub labels: String,

    /// Configuration for the encoders.
    pub encoders: EncodersConfig,
}

/// Model configuration.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    /// Model parameters.
    pub parameters: String,

    /// Configuration of position embeddings.
    pub position_embeddings: PositionEmbeddings,

    /// Configuration of the pretrained model.
    pub pretrain_config: String,

    /// Type of pretraining model.
    pub pretrain_type: PretrainModelType,
}

impl Model {
    /// Read the pretraining model configuration.
    pub fn pretrain_config(&self) -> Result<PretrainConfig, StickerError> {
        let reader = BufReader::new(File::open(&self.pretrain_config)?);

        Ok(match self.pretrain_type {
            PretrainModelType::Albert => {
                PretrainConfig::Albert(serde_json::from_reader(reader).map_err(|err| {
                    StickerError::JSonSerialization(
                        format!(
                            "Cannot read model ALBERT config file `{}`",
                            self.pretrain_config
                        ),
                        err,
                    )
                })?)
            }
            PretrainModelType::Bert => {
                PretrainConfig::Bert(serde_json::from_reader(reader).map_err(|err| {
                    StickerError::JSonSerialization(
                        format!(
                            "Cannot read model BERT config file `{}`",
                            self.pretrain_config
                        ),
                        err,
                    )
                })?)
            }
            PretrainModelType::XlmRoberta => {
                PretrainConfig::XlmRoberta(serde_json::from_reader(reader).map_err(|err| {
                    StickerError::JSonSerialization(
                        format!(
                            "Cannot read model XLM-RoBERTa config file `{}`",
                            self.pretrain_config
                        ),
                        err,
                    )
                })?)
            }
        })
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PositionEmbeddings {
    /// Use position embeddings from the model.
    Model,

    /// Use generated sinusoidal embeddings.
    ///
    /// If normalize is `true`, the sinusoidal embeddings are
    /// normalized using their l2 norm.
    Sinusoidal {
        #[serde(default = "sinusoidal_normalize_default")]
        normalize: bool,
    },
}

fn sinusoidal_normalize_default() -> bool {
    true
}

#[derive(Debug, Deserialize, Serialize)]
pub enum PretrainConfig {
    Albert(AlbertConfig),
    Bert(BertConfig),
    XlmRoberta(BertConfig),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PretrainModelType {
    Albert,
    Bert,
    XlmRoberta,
}

/// Sequence labeler configuration.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Configuration of the input representations.
    pub input: Input,

    /// Configuration of the labeler.
    pub labeler: Labeler,

    /// Configuration of the model.
    pub model: Model,
}

impl Config {
    /// Make configuration paths relative to the configuration file.
    pub fn relativize_paths<P>(&mut self, config_path: P) -> Result<(), StickerError>
    where
        P: AsRef<Path>,
    {
        let config_path = config_path.as_ref();

        *self.input.tokenizer.vocab_mut() =
            relativize_path(config_path, &self.input.tokenizer.vocab())?;
        self.labeler.labels = relativize_path(config_path, &self.labeler.labels)?;
        self.model.parameters = relativize_path(config_path, &self.model.parameters)?;
        self.model.pretrain_config = relativize_path(config_path, &self.model.pretrain_config)?;

        Ok(())
    }

    /// Construct a word piece tokenizer.
    pub fn tokenizer(&self) -> Result<Box<dyn Tokenize>, StickerError> {
        match self.input.tokenizer {
            Tokenizer::Albert { ref vocab } => {
                let spp = SentencePieceProcessor::load(vocab)?;
                Ok(Box::new(AlbertTokenizer::from(spp)))
            }
            Tokenizer::Bert { ref vocab } => {
                let f = File::open(vocab)?;
                let pieces = WordPieces::try_from(BufReader::new(f).lines())?;
                Ok(Box::new(BertTokenizer::new(pieces, "[UNK]")))
            }
            Tokenizer::XlmRoberta { ref vocab } => {
                let spp = SentencePieceProcessor::load(vocab)?;
                Ok(Box::new(XlmRobertaTokenizer::from(spp)))
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum Tokenizer {
    Albert { vocab: String },
    Bert { vocab: String },
    XlmRoberta { vocab: String },
}

impl Tokenizer {
    fn vocab(&self) -> &str {
        use Tokenizer::*;
        match self {
            Albert { vocab } => &vocab,
            Bert { vocab } => &vocab,
            XlmRoberta { vocab } => &vocab,
        }
    }

    fn vocab_mut(&mut self) -> &mut String {
        use Tokenizer::*;
        match self {
            Albert { ref mut vocab } => vocab,
            Bert { ref mut vocab } => vocab,
            XlmRoberta { ref mut vocab } => vocab,
        }
    }
}

pub trait TomlRead
where
    Self: Sized,
{
    fn from_toml_read(read: impl Read) -> Result<Self, StickerError>;
}

impl TomlRead for Config {
    fn from_toml_read(mut read: impl Read) -> Result<Self, StickerError> {
        let mut data = String::new();
        read.read_to_string(&mut data)?;
        let config: Config = toml::from_str(&data)?;
        Ok(config)
    }
}

fn relativize_path(config_path: &Path, filename: &str) -> Result<String, StickerError> {
    if filename.is_empty() {
        return Ok(filename.to_owned());
    }

    let path = Path::new(&filename);

    // Don't touch absolute paths.
    if path.is_absolute() {
        return Ok(filename.to_owned());
    }

    let abs_config_path = config_path.canonicalize()?;
    Ok(abs_config_path
        .parent()
        .ok_or_else(|| {
            StickerError::RelativizePathError(format!(
                "Cannot get parent path of the configuration file: {}",
                abs_config_path.to_string_lossy()
            ))
        })?
        .join(path)
        .to_str()
        .ok_or_else(|| {
            StickerError::RelativizePathError(format!(
                "Cannot cannot convert parent path to string: {}",
                abs_config_path.to_string_lossy()
            ))
        })?
        .to_owned())
}

#[cfg(test)]
mod tests {
    use sticker_encoders::deprel::POSLayer;
    use sticker_encoders::layer::Layer;
    use sticker_encoders::lemma::BackoffStrategy;

    use crate::config::{
        Config, Input, Labeler, Model, PositionEmbeddings, PretrainModelType, Tokenizer, TomlRead,
    };
    use crate::encoders::{DependencyEncoder, EncoderType, EncodersConfig, NamedEncoderConfig};

    #[test]
    fn config() {
        let config =
            Config::from_toml_read(include_bytes!("../testdata/sticker.conf").as_ref()).unwrap();

        assert_eq!(
            config,
            Config {
                input: Input {
                    tokenizer: Tokenizer::Bert {
                        vocab: "bert-base-german-cased-vocab.txt".to_string()
                    },
                },
                labeler: Labeler {
                    labels: "sticker.labels".to_string(),
                    encoders: EncodersConfig(vec![
                        NamedEncoderConfig {
                            name: "dep".to_string(),
                            encoder: EncoderType::Dependency {
                                encoder: DependencyEncoder::RelativePOS(POSLayer::XPos),
                                root_relation: "root".to_string()
                            }
                        },
                        NamedEncoderConfig {
                            name: "lemma".to_string(),
                            encoder: EncoderType::Lemma(BackoffStrategy::Form)
                        },
                        NamedEncoderConfig {
                            name: "pos".to_string(),
                            encoder: EncoderType::Sequence(Layer::XPos)
                        },
                    ]),
                },
                model: Model {
                    parameters: "epoch-99".to_string(),
                    position_embeddings: PositionEmbeddings::Model,
                    pretrain_config: "bert_config.json".to_string(),
                    pretrain_type: PretrainModelType::Bert,
                }
            }
        );
    }
}
