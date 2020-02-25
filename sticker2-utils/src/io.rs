use std::fs::File;

use stdinout::OrExit;
use sticker2::config::{Config, PretrainConfig, TomlRead};
use sticker2::encoders::Encoders;
use sticker2::input::Tokenize;
use sticker2::model::BertModel;
use tch::nn::VarStore;
use tch::Device;

/// Wrapper around different parts of a model.
pub struct Model {
    pub encoders: Encoders,
    pub model: BertModel,
    pub tokenizer: Box<dyn Tokenize>,
    pub vs: VarStore,
}

impl Model {
    /// Load a model on the given device.
    ///
    /// If `freeze` is true, gradient computation is disabled for the
    /// model parameters.
    pub fn load(config_path: &str, device: Device, freeze: bool) -> Self {
        let config = load_config(config_path);
        Self::load_from(config_path, &config.model.parameters, device, freeze)
    }

    /// Load a model on the given device.
    ///
    /// In contrast to `load_model`, this does not load the parameters
    /// specified in the configuration file, but the parameters from
    /// `parameters_path`.
    ///
    /// If `freeze` is true, gradient computation is disabled for the
    /// model parameters.
    pub fn load_from(
        config_path: &str,
        parameters_path: &str,
        device: Device,
        freeze: bool,
    ) -> Model {
        let config = load_config(config_path);
        let encoders = load_encoders(&config);
        let tokenizer = load_tokenizer(&config);
        let pretrain_config = load_pretrain_config(&config);

        let mut vs = VarStore::new(device);

        let model = BertModel::new(
            vs.root(),
            &pretrain_config,
            &encoders,
            0.0,
            config.model.position_embeddings,
        )
        .or_exit("Cannot construct model", 1);

        vs.load(parameters_path)
            .or_exit("Cannot load model parameters", 1);

        if freeze {
            vs.freeze();
        }

        Model {
            encoders,
            model,
            tokenizer,
            vs,
        }
    }

    /// Load a model on the given device.
    ///
    /// In contrast to `load_model`, this does not load the parameters
    /// specified in the configuration file, but the parameters from
    /// the HDF5 file at `hdf5_path`.
    #[cfg(feature = "load-hdf5")]
    pub fn load_from_hdf5(config_path: &str, hdf5_path: &str, device: Device) -> Model {
        let config = load_config(config_path);
        let encoders = load_encoders(&config);
        let tokenizer = load_tokenizer(&config);
        let pretrain_config = load_pretrain_config(&config);

        let vs = VarStore::new(device);

        let model =
            BertModel::from_pretrained(vs.root(), &pretrain_config, hdf5_path, &encoders, 0.5)
                .or_exit("Cannot load pretrained model parameters", 1);

        Model {
            encoders,
            model,
            tokenizer,
            vs,
        }
    }

    #[cfg(not(feature = "load-hdf5"))]
    pub fn load_from_hdf5(_config_path: &str, _hdf5_path: &str, _device: Device) -> Model {
        eprintln!("Cannot load HDF5 model: sticker2 was compiled without support for HDF5");
        std::process::exit(1);
    }
}

pub fn load_pretrain_config(config: &Config) -> PretrainConfig {
    config
        .model
        .pretrain_config()
        .or_exit("Cannot load pretraining model configuration", 1)
}

pub fn load_config(config_path: &str) -> Config {
    let config_file = File::open(config_path).or_exit(
        format!("Cannot open configuration file '{}'", &config_path),
        1,
    );
    let mut config = Config::from_toml_read(config_file).or_exit(
        format!("Cannot parse configuration file: {}", config_path),
        1,
    );
    config.relativize_paths(config_path).or_exit(
        format!(
            "Cannot relativize paths in configuration file: {}",
            config_path
        ),
        1,
    );

    config
}

fn load_encoders(config: &Config) -> Encoders {
    let f = File::open(&config.labeler.labels).or_exit("Cannot open label file", 1);
    let encoders: Encoders = serde_yaml::from_reader(&f).or_exit("Cannot deserialize labels", 1);

    for encoder in &*encoders {
        eprintln!(
            "Loaded labels for encoder '{}': {} labels",
            encoder.name(),
            encoder.encoder().len()
        );
    }

    encoders
}

pub fn load_tokenizer(config: &Config) -> Box<dyn Tokenize> {
    config
        .tokenizer()
        .or_exit("Cannot read tokenizer vocabulary", 1)
}
