use std::io::{BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};
use std::ops::Deref;
use std::sync::Arc;

use clap::{App, Arg, ArgMatches};
use conllx::io::{ReadSentence, Reader, Writer};
use stdinout::OrExit;
use sticker2::input::Tokenize;
use sticker2::tagger::Tagger;
use tch::{self, Device};
use threadpool::ThreadPool;

use crate::io::Model;
use crate::progress::TaggerSpeed;
use crate::sent_proc::SentProcessor;
use crate::traits::{StickerApp, DEFAULT_CLAP_SETTINGS};

const ADDR: &str = "ADDR";
const BATCH_SIZE: &str = "BATCH_SIZE";
const CONFIG: &str = "CONFIG";
const GPU: &str = "GPU";
const MAX_LEN: &str = "MAX_LEN";
const READ_AHEAD: &str = "READ_AHEAD";
const THREADS: &str = "THREADS";

/// A wrapper of `Tagger` that is `Send + Sync`.
///
/// Tensors are not thread-safe in the general case, but
/// multi-threaded use is safe if no (in-place) modifications are
/// made:
///
/// https://discuss.pytorch.org/t/is-evaluating-the-network-thread-safe/37802
struct TaggerWrap(Tagger);

unsafe impl Send for TaggerWrap {}

unsafe impl Sync for TaggerWrap {}

impl Deref for TaggerWrap {
    type Target = Tagger;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone)]
pub struct ServerApp {
    batch_size: usize,
    config: String,
    device: Device,
    addr: String,
    max_len: Option<usize>,
    n_threads: usize,
    read_ahead: usize,
}

impl ServerApp {
    fn serve(
        &self,
        tokenizer: Arc<dyn Tokenize>,
        tagger: Arc<TaggerWrap>,
        pool: ThreadPool,
        listener: TcpListener,
    ) {
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let app = self.clone();
                    let tagger = tagger.clone();
                    let tokenizer = tokenizer.clone();
                    pool.execute(move || handle_client(app, tokenizer, tagger, stream))
                }
                Err(err) => eprintln!("Error processing stream: {}", err),
            }
        }
    }
}

impl StickerApp for ServerApp {
    fn app() -> App<'static, 'static> {
        App::new("server")
            .settings(DEFAULT_CLAP_SETTINGS)
            .about("Annotation server")
            .arg(
                Arg::with_name(CONFIG)
                    .help("Sticker configuration file")
                    .index(1)
                    .required(true),
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
                    .default_value("100"),
            )
            .arg(
                Arg::with_name(THREADS)
                    .short("t")
                    .long("threads")
                    .value_name("N")
                    .help("Number of threads")
                    .default_value("4"),
            )
            .arg(
                Arg::with_name(ADDR)
                    .long("addr")
                    .help("Address to bind to (e.g. localhost:4000)")
                    .default_value("localhost:4000"),
            )
    }

    fn parse(matches: &ArgMatches) -> Self {
        let batch_size = matches
            .value_of(BATCH_SIZE)
            .unwrap()
            .parse()
            .or_exit("Cannot parse batch size", 1);
        let config = matches.value_of(CONFIG).unwrap().to_owned();
        let device = match matches.value_of("GPU") {
            Some(gpu) => Device::Cuda(
                gpu.parse()
                    .or_exit(format!("Cannot parse GPU number ({})", gpu), 1),
            ),
            None => Device::Cpu,
        };
        let addr = matches.value_of(ADDR).unwrap().into();
        let max_len = matches
            .value_of(MAX_LEN)
            .map(|v| v.parse().or_exit("Cannot parse maximum sentence length", 1));
        let n_threads = matches
            .value_of(THREADS)
            .map(|v| v.parse().or_exit("Cannot parse number of threads", 1))
            .unwrap();
        let read_ahead = matches
            .value_of(READ_AHEAD)
            .unwrap()
            .parse()
            .or_exit("Cannot parse number of batches to read ahead", 1);

        ServerApp {
            batch_size,
            addr,
            config,
            device,
            max_len,
            n_threads,
            read_ahead,
        }
    }

    fn run(&self) {
        let model = Model::load(&self.config, self.device, true);
        let tagger = Tagger::new(self.device, model.model, model.encoders);

        let pool = ThreadPool::new(self.n_threads);

        // Set number of PyTorch threads to the number of server
        // threads.  note that this may result in a larger number of
        // threads, depending on libtorch build options. E.g. with
        // OpenMP, each interop thread could create its own intra_op
        // thread pool.
        //
        // If we set the number of Torch threads before creating the
        // Rust thread pool, using one threads will use all CPUs :(.
        tch::set_num_threads(self.n_threads as i32);
        tch::set_num_interop_threads(self.n_threads as i32);

        let listener =
            TcpListener::bind(&self.addr).or_exit(format!("Cannot listen on '{}'", self.addr), 1);

        self.serve(
            Arc::from(model.tokenizer),
            Arc::new(TaggerWrap(tagger)),
            pool,
            listener,
        );
    }
}

fn handle_client(
    app: ServerApp,
    tokenizer: Arc<dyn Tokenize>,
    tagger: Arc<TaggerWrap>,
    mut stream: TcpStream,
) {
    let peer_addr = stream
        .peer_addr()
        .map(|addr| addr.to_string())
        .unwrap_or_else(|_| "<unknown>".to_string());
    eprintln!("Accepted connection from {}", peer_addr);

    let conllx_stream = match stream.try_clone() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("Cannot clone stream: {}", err);
            return;
        }
    };

    let reader = Reader::new(BufReader::new(&conllx_stream));
    let writer = Writer::new(BufWriter::new(&conllx_stream));

    let mut speed = TaggerSpeed::new();

    let mut sent_proc = SentProcessor::new(
        &*tokenizer,
        &tagger,
        writer,
        app.batch_size,
        app.max_len,
        app.read_ahead,
    );

    for sentence in reader.sentences() {
        let sentence = match sentence {
            Ok(sentence) => sentence,
            Err(err) => {
                let _ = writeln!(stream, "! Cannot parse sentence: {}", err);
                return;
            }
        };
        if let Err(err) = sent_proc.process(sentence) {
            let _ = writeln!(stream, "! Error processing sentence: {}", err);
            return;
        }

        speed.count_sentence()
    }

    eprintln!("Finished processing for {}", peer_addr);
}
