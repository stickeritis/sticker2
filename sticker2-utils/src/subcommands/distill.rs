use std::cell::RefCell;
use std::collections::btree_map::{BTreeMap, Entry};
use std::fs::File;
use std::io::{BufReader, Read, Seek};

use clap::{App, Arg, ArgMatches};
use failure::{Fallible, ResultExt};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use ordered_float::NotNan;
use stdinout::OrExit;
use sticker2::config::Config;
use sticker2::dataset::{ConllxDataSet, DataSet};
use sticker2::encoders::Encoders;
use sticker2::input::WordPieceTokenizer;
use sticker2::lr::{ExponentialDecay, LearningRateSchedule};
use sticker2::model::BertModel;
use sticker2::optimizers::{AdamW, AdamWConfig};
use sticker2::tensor::Tensors;
use sticker2::util::seq_len_to_mask;
use tch::nn::VarStore;
use tch::{self, Device, Kind, Reduction, Tensor};

use crate::io::{load_config, Model};
use crate::progress::ReadProgress;
use crate::traits::{StickerApp, DEFAULT_CLAP_SETTINGS};
use crate::util::count_conllx_sentences;

const BATCH_SIZE: &str = "BATCH_SIZE";
const EPOCHS: &str = "EPOCHS";
const EVAL_STEPS: &str = "EVAL_STEPS";
const TEACHER_CONFIG: &str = "TEACHER_CONFIG";
const STUDENT_CONFIG: &str = "STUDENT_CONFIG";
const GPU: &str = "GPU";
const HARD_LOSS: &str = "HARD_LOSS";
const INITIAL_LR_CLASSIFIER: &str = "INITIAL_LR_CLASSIFIER";
const INITIAL_LR_ENCODER: &str = "INITIAL_LR_ENCODER";
const LR_DECAY_RATE: &str = "LR_DECAY_RATE";
const LR_DECAY_STEPS: &str = "LR_DECAY_STEPS";
const MAX_LEN: &str = "MAX_LEN";
const STEPS: &str = "N_STEPS";
const TRAIN_DATA: &str = "TRAIN_DATA";
const VALIDATION_DATA: &str = "VALIDATION_DATA";
const WARMUP: &str = "WARMUP";
const WEIGHT_DECAY: &str = "WEIGHT_DECAY";

pub struct DistillApp {
    batch_size: usize,
    device: Device,
    eval_steps: usize,
    hard_loss: bool,
    max_len: Option<usize>,
    lr_schedules: RefCell<LearningRateSchedules>,
    student_config: String,
    teacher_config: String,
    train_data: String,
    train_duration: TrainDuration,
    validation_data: String,
    weight_decay: f64,
}

pub struct LearningRateSchedules {
    pub classifier: ExponentialDecay,
    pub encoder: ExponentialDecay,
}

struct StudentModel {
    inner: BertModel,
    vs: VarStore,
}

impl DistillApp {
    fn distill_model(
        &self,
        optimizer: &mut AdamW,
        teacher: &Model,
        student: &StudentModel,
        train_file: &File,
        validation_file: &mut File,
    ) -> Fallible<()> {
        let mut best_step = 0;
        let mut best_acc = 0.0;

        let mut global_step = 0;

        let n_steps = self
            .train_duration
            .to_steps(&train_file, self.batch_size)
            .or_exit("Cannot determine number of training steps", 1);

        let train_progress = ProgressBar::new(n_steps as u64);
        train_progress.set_style(ProgressStyle::default_bar().template(
            "[Time: {elapsed_precise}, ETA: {eta_precise}] {bar} {percent}% train {msg}",
        ));

        while global_step < n_steps - 1 {
            let mut train_dataset = Self::open_dataset(&train_file);

            let train_batches = train_dataset.batches(
                &teacher.encoders,
                &teacher.tokenizer,
                self.batch_size,
                self.max_len,
                None,
                false,
            )?;

            for steps in &train_batches.chunks(self.eval_steps) {
                self.train_steps(
                    &train_progress,
                    steps,
                    &mut global_step,
                    optimizer,
                    &teacher.model,
                    &student.inner,
                )?;

                let acc = tch::no_grad(|| {
                    self.validation_epoch(
                        &teacher.encoders,
                        &teacher.tokenizer,
                        &student.inner,
                        validation_file,
                        global_step,
                    )
                })?;

                if acc > best_acc {
                    best_step = global_step;
                    best_acc = acc;

                    student
                        .vs
                        .save(format!("distill-step-{}", global_step))
                        .context(format!(
                            "Cannot save variable store for step {}",
                            global_step
                        ))?;
                }

                let step_status = if best_step == global_step { "🎉" } else { "" };

                eprintln!(
                    "Step {} (validation): acc: {:.4}, best step: {}, best acc: {:.4} {}\n",
                    global_step, acc, best_step, best_acc, step_status
                );

                if global_step >= n_steps - 1 {
                    break;
                }
            }
        }

        Ok(())
    }

    fn train_steps(
        &self,
        progress: &ProgressBar,
        batches: impl Iterator<Item = Fallible<Tensors>>,
        global_step: &mut usize,
        optimizer: &mut AdamW,
        teacher: &BertModel,
        student: &BertModel,
    ) -> Fallible<()> {
        for batch in batches {
            let batch = batch.or_exit("Cannot read batch", 1);

            // Compute masks.
            let attention_mask =
                seq_len_to_mask(&batch.seq_lens, batch.inputs.size()[1]).to_device(self.device);
            let token_mask = batch.token_mask.to_kind(Kind::Float).to_device(self.device);

            // Number of tokens.
            let n_tokens = token_mask.sum(Kind::Float);

            let teacher_logits = tch::no_grad(|| {
                teacher.logits(
                    &batch.inputs.to_device(self.device),
                    &attention_mask,
                    false,
                    true,
                    true,
                )
            });

            let student_logits = student.logits(
                &batch.inputs.to_device(self.device),
                &attention_mask,
                true,
                false,
                false,
            );

            let mut soft_loss = Tensor::zeros(&[], (Kind::Float, self.device));
            let mut hard_loss = Tensor::zeros(&[], (Kind::Float, self.device));

            for (encoder_name, teacher_logits) in teacher_logits {
                // Compute the soft loss.
                let student_logits = &student_logits[&encoder_name];
                let teacher_probs = teacher_logits.softmax(-1, Kind::Float);
                let student_logprobs = student_logits.log_softmax(-1, Kind::Float);
                let soft_losses = -(&teacher_probs * &student_logprobs);
                soft_loss += (soft_losses * &token_mask.unsqueeze(-1)).sum(Kind::Float) / &n_tokens;

                if self.hard_loss {
                    let teacher_predictions = teacher_logits.argmax(-1, false);

                    let targets_shape = teacher_logits.size();
                    let batch_size = targets_shape[0];
                    let seq_len = targets_shape[1];

                    let hard_losses = student_logprobs
                        .view([batch_size * seq_len, -1])
                        .g_nll_loss::<&Tensor>(
                            &teacher_predictions.view([batch_size * seq_len]),
                            None,
                            Reduction::None,
                            -100,
                        )
                        .view([batch_size, seq_len]);

                    hard_loss += (hard_losses * &token_mask).sum(Kind::Float) / &n_tokens;
                }
            }

            let loss = &hard_loss + &soft_loss;

            let lr_classifier = self
                .lr_schedules
                .borrow_mut()
                .classifier
                .compute_step_learning_rate(*global_step);
            let lr_encoder = self
                .lr_schedules
                .borrow_mut()
                .encoder
                .compute_step_learning_rate(*global_step);

            optimizer.backward_step(&loss, |name| {
                let mut config = AdamWConfig::default();

                // Use weight decay for all variables, except for
                // layer norm variables.
                if !name.contains("layer_norm") {
                    config.weight_decay = self.weight_decay;
                }

                // Use discriminative learning rates, do not optimize
                // position embeddings.
                if name.contains("position_embeddings") {
                    config.lr = 0.;
                } else if name.starts_with("classifiers") {
                    config.lr = lr_classifier.into();
                } else if name.starts_with("encoder") {
                    config.lr = lr_encoder.into();
                } else {
                    unreachable!();
                }

                config
            });

            progress.set_message(&format!(
                "step: {}, lr encoder: {:.6}, lr classifier: {:.6}, hard loss: {:.4}, soft loss: {:.4}",
                global_step,
                lr_encoder,
                lr_classifier,
                f32::from(hard_loss),
                f32::from(soft_loss)
            ));
            progress.inc(1);

            *global_step += 1;
        }

        Ok(())
    }

    fn open_dataset(file: &File) -> ConllxDataSet<impl Read + Seek> {
        let read = BufReader::new(
            file.try_clone()
                .or_exit("Cannot open data set for reading", 1),
        );
        ConllxDataSet::new(read)
    }

    fn fresh_student(&self, student_config: &Config, teacher: &Model) -> StudentModel {
        let bert_config = student_config
            .model
            .pretrain_config()
            .or_exit("Cannot load pretraining model configuration", 1);

        let vs = VarStore::new(self.device);

        let inner = BertModel::new(
            vs.root(),
            &bert_config,
            &teacher.encoders,
            0.1,
            student_config.model.position_embeddings.clone(),
        )
        .or_exit("Cannot construct fresh student model", 1);

        StudentModel { inner, vs }
    }

    pub fn create_lr_schedules(
        initial_lr_classifier: NotNan<f32>,
        initial_lr_encoder: NotNan<f32>,
        lr_decay_rate: NotNan<f32>,
        lr_decay_steps: usize,
        warmup_steps: usize,
    ) -> LearningRateSchedules {
        let classifier = ExponentialDecay::new(
            initial_lr_classifier.into_inner(),
            lr_decay_rate.into_inner(),
            lr_decay_steps,
            false,
            warmup_steps,
        );

        let mut encoder = classifier.clone();
        encoder.set_initial_lr(initial_lr_encoder.into_inner());

        LearningRateSchedules {
            classifier,
            encoder,
        }
    }

    fn validation_epoch(
        &self,
        encoders: &Encoders,
        tokenizer: &WordPieceTokenizer,
        model: &BertModel,
        file: &mut File,
        global_step: usize,
    ) -> Fallible<f32> {
        let read_progress = ReadProgress::new(file).or_exit("Cannot create progress bar", 1);
        let progress_bar = read_progress.progress_bar().clone();
        progress_bar.set_style(ProgressStyle::default_bar().template(
            "[Time: {elapsed_precise}, ETA: {eta_precise}] {bar} {percent}% validation {msg}",
        ));

        let mut dataset = ConllxDataSet::new(read_progress);

        let mut encoder_accuracy = BTreeMap::new();
        let mut encoder_loss = BTreeMap::new();

        let mut n_tokens = 0;

        for batch in dataset.batches(
            encoders,
            tokenizer,
            self.batch_size,
            self.max_len,
            None,
            true,
        )? {
            let batch = batch?;

            let attention_mask = seq_len_to_mask(&batch.seq_lens, batch.inputs.size()[1]);

            let (summed_loss, encoder_specific_loss, encoder_specific_accuracy) = model.loss(
                &batch.inputs.to_device(self.device),
                &attention_mask.to_device(self.device),
                &batch.token_mask.to_device(self.device),
                &batch
                    .labels
                    .expect("Batch without labels.")
                    .into_iter()
                    .map(|(encoder_name, labels)| (encoder_name, labels.to_device(self.device)))
                    .collect(),
                None,
                false,
                true,
                true,
                false,
            );

            let n_batch_tokens = i64::from(batch.token_mask.sum(Kind::Int64));
            n_tokens += n_batch_tokens;

            let scalar_loss: f32 = summed_loss.sum(Kind::Float).into();

            for (encoder_name, loss) in encoder_specific_loss {
                match encoder_accuracy.entry(encoder_name.clone()) {
                    Entry::Vacant(entry) => {
                        entry.insert(
                            f32::from(&encoder_specific_accuracy[&encoder_name])
                                * n_batch_tokens as f32,
                        );
                    }
                    Entry::Occupied(mut entry) => {
                        *entry.get_mut() += f32::from(&encoder_specific_accuracy[&encoder_name])
                            * n_batch_tokens as f32;
                    }
                };

                match encoder_loss.entry(encoder_name) {
                    Entry::Vacant(entry) => {
                        entry.insert(f32::from(loss) * n_batch_tokens as f32);
                    }
                    Entry::Occupied(mut entry) => {
                        *entry.get_mut() += f32::from(loss) * n_batch_tokens as f32
                    }
                };
            }

            progress_bar.set_message(&format!(
                "batch loss: {:.4}, global step: {}",
                scalar_loss, global_step
            ));
        }

        progress_bar.finish();

        eprintln!();
        let mut acc_sum = 0.0;
        for (encoder_name, loss) in encoder_loss {
            let acc = encoder_accuracy[&encoder_name] / n_tokens as f32;

            eprintln!(
                "{} loss: {} accuracy: {:.4}",
                encoder_name,
                loss / n_tokens as f32,
                acc
            );

            acc_sum += acc;
        }
        eprintln!();

        Ok(acc_sum / encoders.len() as f32)
    }
}

impl StickerApp for DistillApp {
    fn app() -> App<'static, 'static> {
        App::new("distill")
            .settings(DEFAULT_CLAP_SETTINGS)
            .about("Distill a model")
            .arg(
                Arg::with_name(TEACHER_CONFIG)
                    .help("Teacher configuration file")
                    .index(1)
                    .required(true),
            )
            .arg(
                Arg::with_name(STUDENT_CONFIG)
                    .help("Student configuration file")
                    .index(2)
                    .required(true),
            )
            .arg(
                Arg::with_name(TRAIN_DATA)
                    .help("Training data")
                    .index(3)
                    .required(true),
            )
            .arg(
                Arg::with_name(VALIDATION_DATA)
                    .help("Validation data")
                    .index(4)
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
                Arg::with_name(EPOCHS)
                    .long("epochs")
                    .takes_value(true)
                    .value_name("N")
                    .help("Train for N epochs")
                    .default_value("2"),
            )
            .arg(
                Arg::with_name(EVAL_STEPS)
                    .long("eval_steps")
                    .takes_value(true)
                    .value_name("N")
                    .help("Evaluate after N steps, save the model on improvement")
                    .default_value("1000"),
            )
            .arg(
                Arg::with_name(GPU)
                    .long("gpu")
                    .takes_value(true)
                    .help("Use the GPU with the given identifier"),
            )
            .arg(
                Arg::with_name(HARD_LOSS)
                    .long("hard-loss")
                    .help("Add hard loss (predicted label) to the soft loss"),
            )
            .arg(
                Arg::with_name(INITIAL_LR_CLASSIFIER)
                    .long("lr-classifier")
                    .value_name("LR")
                    .help("Initial classifier learning rate")
                    .default_value("1e-3"),
            )
            .arg(
                Arg::with_name(INITIAL_LR_ENCODER)
                    .long("lr-encoder")
                    .value_name("LR")
                    .help("Initial encoder learning rate")
                    .default_value("5e-5"),
            )
            .arg(
                Arg::with_name(LR_DECAY_RATE)
                    .long("lr-decay-rate")
                    .value_name("N")
                    .help("Exponential decay rate")
                    .default_value("0.99998"),
            )
            .arg(
                Arg::with_name(LR_DECAY_STEPS)
                    .long("lr-decay-steps")
                    .value_name("N")
                    .help("Exponential decay rate")
                    .default_value("10"),
            )
            .arg(
                Arg::with_name(MAX_LEN)
                    .long("maxlen")
                    .value_name("N")
                    .takes_value(true)
                    .help("Ignore sentences longer than N tokens"),
            )
            .arg(
                Arg::with_name(STEPS)
                    .long("steps")
                    .value_name("N")
                    .help("Train for N steps")
                    .takes_value(true)
                    .overrides_with(EPOCHS),
            )
            .arg(
                Arg::with_name(WARMUP)
                    .long("warmup")
                    .value_name("N")
                    .help(
                        "For the first N timesteps, the learning rate is linearly scaled up to LR.",
                    )
                    .default_value("2000"),
            )
            .arg(
                Arg::with_name(WEIGHT_DECAY)
                    .long("weight-decay")
                    .value_name("D")
                    .help("Weight decay (L2 penalty).")
                    .default_value("0.0"),
            )
    }

    fn parse(matches: &ArgMatches) -> Self {
        let teacher_config = matches.value_of(TEACHER_CONFIG).unwrap().into();
        let student_config = matches.value_of(STUDENT_CONFIG).unwrap().into();
        let train_data = matches.value_of(TRAIN_DATA).map(ToOwned::to_owned).unwrap();
        let validation_data = matches
            .value_of(VALIDATION_DATA)
            .map(ToOwned::to_owned)
            .unwrap();
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
        let eval_steps = matches
            .value_of(EVAL_STEPS)
            .unwrap()
            .parse()
            .or_exit("Cannot parse number of batches after which to save", 1);
        let hard_loss = matches.is_present(HARD_LOSS);
        let initial_lr_classifier = matches
            .value_of(INITIAL_LR_CLASSIFIER)
            .unwrap()
            .parse()
            .or_exit("Cannot parse initial classifier learning rate", 1);
        let initial_lr_encoder = matches
            .value_of(INITIAL_LR_ENCODER)
            .unwrap()
            .parse()
            .or_exit("Cannot parse initial encoder learning rate", 1);
        let lr_decay_rate = matches
            .value_of(LR_DECAY_RATE)
            .unwrap()
            .parse()
            .or_exit("Cannot parse exponential decay rate", 1);
        let lr_decay_steps = matches
            .value_of(LR_DECAY_STEPS)
            .unwrap()
            .parse()
            .or_exit("Cannot parse exponential decay steps", 1);
        let max_len = matches
            .value_of(MAX_LEN)
            .map(|v| v.parse().or_exit("Cannot parse maximum sentence length", 1));
        let warmup_steps = matches
            .value_of(WARMUP)
            .unwrap()
            .parse()
            .or_exit("Cannot parse warmup", 1);
        let weight_decay = matches
            .value_of(WEIGHT_DECAY)
            .unwrap()
            .parse()
            .or_exit("Cannot parse weight decay", 1);

        // If steps is present, it overrides epochs.
        let train_duration = if let Some(steps) = matches.value_of(STEPS) {
            let steps = steps
                .parse()
                .or_exit("Cannot parse the number of training steps", 1);
            TrainDuration::Steps(steps)
        } else {
            let epochs = matches
                .value_of(EPOCHS)
                .unwrap()
                .parse()
                .or_exit("Cannot parse number of training epochs", 1);
            TrainDuration::Epochs(epochs)
        };

        DistillApp {
            batch_size,
            device,
            eval_steps,
            hard_loss,
            max_len,
            lr_schedules: RefCell::new(Self::create_lr_schedules(
                initial_lr_classifier,
                initial_lr_encoder,
                lr_decay_rate,
                lr_decay_steps,
                warmup_steps,
            )),
            student_config,
            teacher_config,
            train_data,
            train_duration,
            validation_data,
            weight_decay,
        }
    }

    fn run(&self) {
        let student_config = load_config(&self.student_config);
        let teacher = Model::load(&self.teacher_config, self.device, true);

        let train_file = File::open(&self.train_data).or_exit("Cannot open train data file", 1);
        let mut validation_file =
            File::open(&self.validation_data).or_exit("Cannot open validation data file", 1);

        let student = self.fresh_student(&student_config, &teacher);

        let mut optimizer = AdamW::new(&student.vs);

        self.distill_model(
            &mut optimizer,
            &teacher,
            &student,
            &train_file,
            &mut validation_file,
        )
        .or_exit("Model distillation failed", 1);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TrainDuration {
    Epochs(usize),
    Steps(usize),
}

impl TrainDuration {
    fn to_steps(&self, train_file: &File, batch_size: usize) -> Fallible<usize> {
        use TrainDuration::*;

        match *self {
            Epochs(epochs) => {
                eprintln!("Counting number of steps in an epoch...");
                let read_progress =
                    ReadProgress::new(train_file.try_clone()?).or_exit("Cannot open train file", 1);

                let progress_bar = read_progress.progress_bar().clone();
                progress_bar
                    .set_style(ProgressStyle::default_bar().template(
                        "[Time: {elapsed_precise}, ETA: {eta_precise}] {bar} {percent}%",
                    ));

                let n_sentences = count_conllx_sentences(BufReader::new(read_progress))?;

                progress_bar.finish_and_clear();

                // Compute number of steps of the given batch size.
                let steps_per_epoch = (n_sentences + batch_size - 1) / batch_size;
                eprintln!(
                    "sentences: {}, steps_per epoch: {}, total_steps: {}",
                    n_sentences,
                    steps_per_epoch,
                    epochs * steps_per_epoch
                );
                Ok(epochs * steps_per_epoch)
            }
            Steps(steps) => Ok(steps),
        }
    }
}
