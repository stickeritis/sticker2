use std::fs::File;
use std::io::Write;

use clap::{App, Arg, ArgMatches};
use conllx::io::{ReadSentence, Reader};
use serde_yaml;
use stdinout::{Input, OrExit};
use sticker2::config::Config;
use sticker2::encoders::Encoders;
use sticker_encoders::SentenceEncoder;

use crate::io::load_config;
use crate::traits::{StickerApp, DEFAULT_CLAP_SETTINGS};

const CONFIG: &str = "CONFIG";
static TRAIN_DATA: &str = "TRAIN_DATA";

pub struct PrepareApp {
    config: String,
    train_data: Option<String>,
}

impl PrepareApp {
    fn write_labels(config: &Config, encoders: &Encoders) {
        let mut f = File::create(&config.labeler.labels).or_exit("Cannot create label file", 1);
        let serialized_labels =
            serde_yaml::to_string(&encoders).or_exit("Cannot serialize labels", 1);
        f.write_all(serialized_labels.as_bytes())
            .or_exit("Cannot write labels", 1);
    }
}

impl StickerApp for PrepareApp {
    fn app() -> App<'static, 'static> {
        App::new("prepare")
            .settings(DEFAULT_CLAP_SETTINGS)
            .about("Prepare shape and label files for training")
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

        PrepareApp { config, train_data }
    }

    fn run(&self) {
        let config = load_config(&self.config);

        let encoders: Encoders = (&config.labeler.encoders).into();

        let input = Input::from(self.train_data.as_ref());
        let treebank_reader = Reader::new(
            input
                .buf_read()
                .or_exit("Cannot open corpus for reading", 1),
        );

        for sentence in treebank_reader.sentences() {
            let sentence = sentence.or_exit("Cannot read sentence from treebank", 1);

            for encoder in &*encoders {
                encoder.encoder().encode(&sentence).or_exit(
                    format!("Cannot encode sentence with encoder {}", encoder.name()),
                    1,
                );
            }
        }

        Self::write_labels(&config, &encoders);
    }
}
