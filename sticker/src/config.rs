use std::io::Read;
use std::path::Path;

use failure::{format_err, Fallible};
use serde::{Deserialize, Serialize};
use toml;

use crate::encoders::EncodersConfig;

/// Labeler configuration.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Labeler {
    /// The encoder labels file.
    pub labels: String,

    /// Configuration for the encoders.
    pub encoders: EncodersConfig,
}

/// Sequence labeler configuration.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Configuration of the labeler.
    pub labeler: Labeler,
}

impl Config {
    /// Make configuration paths relative to the configuration file.
    pub fn relativize_paths<P>(&mut self, config_path: P) -> Fallible<()>
    where
        P: AsRef<Path>,
    {
        let config_path = config_path.as_ref();

        self.labeler.labels = relativize_path(config_path, &self.labeler.labels)?;

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

    use maplit::hashmap;
    use sticker_encoders::layer::Layer;
    use sticker_encoders::lemma::BackoffStrategy;

    use crate::config::{Config, Labeler, TomlRead};
    use crate::encoders::{DependencyEncoder, EncoderType, EncodersConfig};

    #[test]
    fn config() {
        let config_file = File::open("testdata/sticker.conf").unwrap();
        let config = Config::from_toml_read(config_file).unwrap();

        assert_eq!(
            config,
            Config {
                labeler: Labeler {
                    labels: "sticker.labels".to_string(),
                    encoders: EncodersConfig(hashmap![
                    "dep".to_string() => EncoderType::Dependency(DependencyEncoder::RelativePOS),
                    "lemma".to_string() => EncoderType::Lemma(BackoffStrategy::Form),
                    "pos".to_string() => EncoderType::Sequence(Layer::Pos),
                            ]),
                }
            }
        );
    }
}
