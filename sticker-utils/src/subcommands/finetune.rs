use std::fs::File;

use clap::{App, Arg, ArgMatches};
use conllx::io::{ReadSentence, Reader};
use serde_yaml;
use stdinout::{Input, OrExit};
use sticker::config::{Config, TomlRead};
use sticker::encoders::Encoders;

use crate::traits::{StickerApp, DEFAULT_CLAP_SETTINGS};

const CONFIG: &str = "CONFIG";
static TRAIN_DATA: &str = "TRAIN_DATA";

pub struct FinetuneApp {
    config: String,
    train_data: Option<String>,
}

impl StickerApp for FinetuneApp {
    fn app() -> App<'static, 'static> {
        App::new("finetune")
            .settings(DEFAULT_CLAP_SETTINGS)
            .about("Finetune a model")
            .arg(
                Arg::with_name(CONFIG)
                    .help("Sticker configuration file")
                    .index(1)
                    .required(true),
            )
            .arg(Arg::with_name(TRAIN_DATA).help("Training data").index(2))
    }

    fn parse(matches: &ArgMatches) -> Self {
        let config = matches.value_of(CONFIG).unwrap().into();
        let train_data = matches.value_of(TRAIN_DATA).map(ToOwned::to_owned);

        FinetuneApp { config, train_data }
    }

    fn run(&self) {
        let config_file = File::open(&self.config).or_exit(
            format!("Cannot open configuration file '{}'", &self.config),
            1,
        );
        let mut config =
            Config::from_toml_read(config_file).or_exit("Cannot parse configuration", 1);
        config
            .relativize_paths(&self.config)
            .or_exit("Cannot relativize paths in configuration", 1);

        let input = Input::from(self.train_data.as_ref());
        let treebank_reader = Reader::new(
            input
                .buf_read()
                .or_exit("Cannot open corpus for reading", 1),
        );

        let f = File::open(&config.labeler.labels).or_exit("Cannot open label file", 1);
        let encoders: Encoders =
            serde_yaml::from_reader(&f).or_exit("Cannot deserialize labels", 1);

        for encoder_name in encoders.keys() {
            eprintln!("Loaded labels for encoder '{}'", encoder_name);
        }

        for sentence in treebank_reader.sentences() {
            let _sentence = sentence.or_exit("Cannot read sentence from treebank", 1);
        }
    }
}
