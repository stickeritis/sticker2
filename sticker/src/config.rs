use std::convert::TryFrom;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use failure::{format_err, Fallible};
use serde::{Deserialize, Serialize};
use serde_json;
use sticker_transformers::models::bert::BertConfig;
use toml;
use wordpieces::WordPieces;

use crate::encoders::EncodersConfig;
use crate::input::vectorizer::{ReadWordPieces, WordPieceVectorizer};
use crate::input::WordPieceTokenizer;

/// Input configuration.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    /// Word pieces file.
    pub word_pieces: String,
}

impl Input {
    /// Construct a word piece tokenizer.
    pub fn word_piece_tokenizer(&self) -> Fallible<WordPieceTokenizer> {
        let f = File::open(&self.word_pieces)?;
        let pieces = WordPieces::try_from(BufReader::new(f).lines())?;
        Ok(WordPieceTokenizer::new(pieces, "[UNK]"))
    }

    /// Construct a word piece vectorizer.
    pub fn word_piece_vectorizer(&self) -> Fallible<WordPieceVectorizer> {
        let f = File::open(&self.word_pieces)?;
        Ok(WordPieceVectorizer::read_word_pieces(BufReader::new(f))?)
    }
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

    /// Configuration of the pretrained model.
    pub pretrain_config: String,
}

impl Model {
    /// Read the pretraining model configuration.
    pub fn pretrain_config(&self) -> Fallible<BertConfig> {
        let f = File::open(&self.pretrain_config)?;
        Ok(serde_json::from_reader(BufReader::new(f))?)
    }
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
    pub fn relativize_paths<P>(&mut self, config_path: P) -> Fallible<()>
    where
        P: AsRef<Path>,
    {
        let config_path = config_path.as_ref();

        self.input.word_pieces = relativize_path(config_path, &self.input.word_pieces)?;
        self.labeler.labels = relativize_path(config_path, &self.labeler.labels)?;
        self.model.parameters = relativize_path(config_path, &self.model.parameters)?;
        self.model.pretrain_config = relativize_path(config_path, &self.model.pretrain_config)?;

        Ok(())
    }
}

pub trait TomlRead
where
    Self: Sized,
{
    fn from_toml_read(read: impl Read) -> Fallible<Self>;
}

impl TomlRead for Config {
    fn from_toml_read(mut read: impl Read) -> Fallible<Self> {
        let mut data = String::new();
        read.read_to_string(&mut data)?;
        let config: Config = toml::from_str(&data)?;
        Ok(config)
    }
}

fn relativize_path(config_path: &Path, filename: &str) -> Fallible<String> {
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
            format_err!(
                "Cannot get parent path of the configuration file: {}",
                abs_config_path.to_string_lossy()
            )
        })?
        .join(path)
        .to_str()
        .ok_or_else(|| {
            format_err!(
                "Cannot cannot convert partent path to string: {}",
                abs_config_path.to_string_lossy()
            )
        })?
        .to_owned())
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use sticker_encoders::layer::Layer;
    use sticker_encoders::lemma::BackoffStrategy;

    use crate::config::{Config, Input, Labeler, Model, TomlRead};
    use crate::encoders::{DependencyEncoder, EncoderType, EncodersConfig, NamedEncoderConfig};

    #[test]
    fn config() {
        let config_file = File::open("testdata/sticker.conf").unwrap();
        let config = Config::from_toml_read(config_file).unwrap();

        assert_eq!(
            config,
            Config {
                input: Input {
                    word_pieces: "bert-base-german-cased-vocab.txt".to_string()
                },
                labeler: Labeler {
                    labels: "sticker.labels".to_string(),
                    encoders: EncodersConfig(vec![
                        NamedEncoderConfig {
                            name: "dep".to_string(),
                            encoder: EncoderType::Dependency(DependencyEncoder::RelativePOS)
                        },
                        NamedEncoderConfig {
                            name: "lemma".to_string(),
                            encoder: EncoderType::Lemma(BackoffStrategy::Form)
                        },
                        NamedEncoderConfig {
                            name: "pos".to_string(),
                            encoder: EncoderType::Sequence(Layer::Pos)
                        },
                    ]),
                },
                model: Model {
                    parameters: "epoch-99".to_string(),
                    pretrain_config: "bert_config.json".to_string(),
                }
            }
        );
    }
}
