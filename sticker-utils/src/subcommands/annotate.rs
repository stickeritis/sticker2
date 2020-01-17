use std::io::BufWriter;

use clap::{App, Arg, ArgMatches};
use conllx::io::{ReadSentence, Reader, WriteSentence, Writer};
use stdinout::{Input, OrExit, Output};
use sticker::input::WordPieceTokenizer;
use sticker::tagger::Tagger;
use tch::{self, Device};

use crate::io::Model;
use crate::progress::TaggerSpeed;
use crate::sent_proc::SentProcessor;
use crate::traits::{StickerApp, DEFAULT_CLAP_SETTINGS};

const BATCH_SIZE: &str = "BATCH_SIZE";
const CONFIG: &str = "CONFIG";
const GPU: &str = "GPU";
const INPUT: &str = "INPUT";
const MAX_LEN: &str = "MAX_LEN";
const OUTPUT: &str = "OUTPUT";
const READ_AHEAD: &str = "READ_AHEAD";

pub struct AnnotateApp {
    batch_size: usize,
    config: String,
    device: Device,
    input: Option<String>,
    max_len: Option<usize>,
    output: Option<String>,
    read_ahead: usize,
}

impl AnnotateApp {
    fn process<R, W>(&self, tokenizer: &WordPieceTokenizer, tagger: Tagger, read: R, write: W)
    where
        R: ReadSentence,
        W: WriteSentence,
    {
        let mut speed = TaggerSpeed::new();

        let mut sent_proc = SentProcessor::new(
            &tokenizer,
            &tagger,
            write,
            self.batch_size,
            self.max_len,
            self.read_ahead,
        );

        for sentence in read.sentences() {
            let sentence = sentence.or_exit("Cannot parse sentence", 1);
            sent_proc
                .process(sentence)
                .or_exit("Error processing sentence", 1);

            speed.count_sentence()
        }
    }
}

impl StickerApp for AnnotateApp {
    fn app() -> App<'static, 'static> {
        App::new("annotate")
            .settings(DEFAULT_CLAP_SETTINGS)
            .about("Annotate a corpus")
            .arg(
                Arg::with_name(CONFIG)
                    .help("Sticker configuration file")
                    .index(1)
                    .required(true),
            )
            .arg(
                Arg::with_name(INPUT)
                    .help("Input data")
                    .long("input")
                    .takes_value(true),
            )
            .arg(
                Arg::with_name(OUTPUT)
                    .help("Output data")
                    .long("output")
                    .takes_value(true),
            )
            .arg(
                Arg::with_name(BATCH_SIZE)
                    .long("batch-size")
                    .takes_value(true)
                    .help("Batch size")
                    .default_value("32"),
            )
            .arg(
                Arg::with_name(GPU)
                    .long("gpu")
                    .takes_value(true)
                    .help("Use the GPU with the given identifier"),
            )
            .arg(
                Arg::with_name(MAX_LEN)
                    .long("maxlen")
                    .value_name("N")
                    .takes_value(true)
                    .help("Ignore sentences longer than N tokens"),
            )
            .arg(
                Arg::with_name(READ_AHEAD)
                    .help("Readahead (number of batches)")
                    .long("readahead")
                    .default_value("10"),
            )
    }

    fn parse(matches: &ArgMatches) -> Self {
        let config = matches.value_of(CONFIG).unwrap().into();
        let batch_size = matches
            .value_of(BATCH_SIZE)
            .unwrap()
            .parse()
            .or_exit("Cannot parse batch size", 1);
        let device = match matches.value_of("GPU") {
            Some(gpu) => Device::Cuda(
                gpu.parse()
                    .or_exit(format!("Cannot parse GPU number ({})", gpu), 1),
            ),
            None => Device::Cpu,
        };
        let input = matches.value_of(INPUT).map(ToOwned::to_owned);
        let max_len = matches
            .value_of(MAX_LEN)
            .map(|v| v.parse().or_exit("Cannot parse maximum sentence length", 1));
        let output = matches.value_of(OUTPUT).map(ToOwned::to_owned);
        let read_ahead = matches
            .value_of(READ_AHEAD)
            .unwrap()
            .parse()
            .or_exit("Cannot parse number of batches to read ahead", 1);

        AnnotateApp {
            batch_size,
            config,
            device,
            input,
            max_len,
            output,
            read_ahead,
        }
    }

    fn run(&self) {
        let model = Model::load(&self.config, self.device, true);
        let tagger = Tagger::new(self.device, model.model, model.encoders, model.vectorizer);

        let input = Input::from(self.input.as_ref());
        let reader = Reader::new(input.buf_read().or_exit("Cannot open input for reading", 1));

        let output = Output::from(self.output.as_ref());
        let writer = Writer::new(BufWriter::new(
            output.write().or_exit("Cannot open output for writing", 1),
        ));

        self.process(&model.tokenizer, tagger, reader, writer)
    }
}
