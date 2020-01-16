use std::fs::File;

use stdinout::OrExit;
use sticker::config::{Config, TomlRead};
use sticker::encoders::Encoders;
use sticker::input::WordPieceTokenizer;
use sticker::model::BertModel;
use sticker::tagger::Tagger;
use tch::nn::VarStore;
use tch::Device;

pub fn load_tagger(config_path: &str, device: Device) -> (WordPieceTokenizer, Tagger) {
    let config_file = File::open(config_path).or_exit(
        format!("Cannot open configuration file '{}'", &config_path),
        1,
    );
    let mut config = Config::from_toml_read(config_file).or_exit("Cannot parse configuration", 1);
    config
        .relativize_paths(config_path)
        .or_exit("Cannot relativize paths in configuration", 1);

    let f = File::open(&config.labeler.labels).or_exit("Cannot open label file", 1);
    let encoders: Encoders = serde_yaml::from_reader(&f).or_exit("Cannot deserialize labels", 1);

    for encoder in &*encoders {
        eprintln!(
            "Loaded labels for encoder '{}': {} labels",
            encoder.name(),
            encoder.encoder().len()
        );
    }

    let tokenizer = config
        .input
        .word_piece_tokenizer()
        .or_exit("Cannot read word pieces", 1);

    let vectorizer = config
        .input
        .word_piece_vectorizer()
        .or_exit("Cannot read word pieces", 1);

    let bert_config = config
        .model
        .pretrain_config()
        .or_exit("Cannot load pretraining model configuration", 1);

    let mut vs = VarStore::new(device);

    let model = BertModel::new(vs.root(), &bert_config, &encoders, 0.0)
        .or_exit("Cannot construct model", 1);

    vs.load(&config.model.parameters)
        .or_exit("Cannot load model parameters", 1);

    vs.freeze();

    (tokenizer, Tagger::new(device, model, encoders, vectorizer))
}
