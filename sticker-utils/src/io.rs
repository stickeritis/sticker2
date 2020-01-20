use std::fs::File;

use stdinout::OrExit;
use sticker::config::{Config, TomlRead};
use sticker::encoders::Encoders;
use sticker::input::vectorizer::WordPieceVectorizer;
use sticker::input::WordPieceTokenizer;
use sticker::model::BertModel;
use sticker_transformers::models::bert::BertConfig;
use tch::nn::VarStore;
use tch::Device;

/// Wrapper around different parts of a model.
pub struct Model {
    pub encoders: Encoders,
    pub model: BertModel,
    pub tokenizer: WordPieceTokenizer,
    pub vectorizer: WordPieceVectorizer,
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
        let (tokenizer, vectorizer) = load_tokenizer(&config);
        let bert_config = load_bert_config(&config);

        let mut vs = VarStore::new(device);

        let model = BertModel::new(
            vs.root(),
            &bert_config,
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
            vectorizer,
            vs,
        }
    }

    /// Load a model on the given device.
    ///
    /// In contrast to `load_model`, this does not load the parameters
    /// specified in the configuration file, but the parameters from
    /// the HDF5 file at `hdf5_path`.
    pub fn load_from_hdf5(config_path: &str, hdf5_path: &str, device: Device) -> Model {
        let config = load_config(config_path);
        let encoders = load_encoders(&config);
        let (tokenizer, vectorizer) = load_tokenizer(&config);
        let bert_config = load_bert_config(&config);

        let vs = VarStore::new(device);

        let model = BertModel::from_pretrained(vs.root(), &bert_config, hdf5_path, &encoders, 0.5)
            .or_exit("Cannot load pretrained model parameters", 1);

        Model {
            encoders,
            model,
            tokenizer,
            vectorizer,
            vs,
        }
    }
}

fn load_bert_config(config: &Config) -> BertConfig {
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

fn load_tokenizer(config: &Config) -> (WordPieceTokenizer, WordPieceVectorizer) {
    let tokenizer = config
        .input
        .word_piece_tokenizer()
        .or_exit("Cannot read word pieces", 1);

    let vectorizer = config
        .input
        .word_piece_vectorizer()
        .or_exit("Cannot read word pieces", 1);

    (tokenizer, vectorizer)
}
