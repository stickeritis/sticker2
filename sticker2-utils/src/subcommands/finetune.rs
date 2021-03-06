use std::collections::BTreeMap;
use std::fs::File;

use anyhow::{Context, Result};
use clap::{App, Arg, ArgMatches};
use indicatif::ProgressStyle;
use ordered_float::NotNan;
use sticker2::dataset::{ConlluDataSet, DataSet, SequenceLength};
use sticker2::encoders::Encoders;
use sticker2::input::Tokenize;
use sticker2::lr::{ExponentialDecay, LearningRateSchedule, PlateauLearningRate};
use sticker2::model::bert::{BertModel, FreezeLayers};
use sticker2::optimizers::{AdamW, AdamWConfig, GradScaler, Optimizer};
use sticker2::util::seq_len_to_mask;
use tch::{self, Device, Kind};

use crate::io::Model;
use crate::progress::ReadProgress;
use crate::save::{BestEpochSaver, CompletedUnit, Save};
use crate::summary::{SummaryOption, SummaryWriter};
use crate::traits::{StickerApp, StickerOption, DEFAULT_CLAP_SETTINGS};
use crate::util::autocast_or_preserve;

const BATCH_SIZE: &str = "BATCH_SIZE";
const CONFIG: &str = "CONFIG";
const CONTINUE: &str = "CONTINUE";
const GPU: &str = "GPU";
const FINETUNE_EMBEDS: &str = "FINETUNE_EMBEDS";
const INITIAL_LR_CLASSIFIER: &str = "INITIAL_LR_CLASSIFIER";
const INITIAL_LR_ENCODER: &str = "INITIAL_LR_ENCODER";
const LABEL_SMOOTHING: &str = "LABEL_SMOOTHING";
const MIXED_PRECISION: &str = "MIXED_PRECISION";
const INCLUDE_CONTINUATIONS: &str = "INCLUDE_CONTINUATIONS";
const LR_DECAY_RATE: &str = "LR_DECAY_RATE";
const LR_PATIENCE: &str = "LR_PATIENCE";
const LR_SCALE: &str = "LR_SCALE";
const MAX_LEN: &str = "MAX_LEN";
const PATIENCE: &str = "PATIENCE";
const PRETRAINED_MODEL: &str = "PRETRAINED_MODEL";
const TRAIN_DATA: &str = "TRAIN_DATA";
const VALIDATION_DATA: &str = "VALIDATION_DATA";
const WARMUP: &str = "WARMUP";
const WEIGHT_DECAY: &str = "WEIGHT_DECAY";

pub struct LrSchedule {
    pub initial_lr_encoder: NotNan<f32>,
    pub initial_lr_classifier: NotNan<f32>,
    pub lr_decay_rate: NotNan<f32>,
    pub lr_scale: NotNan<f32>,
    pub lr_patience: usize,
    pub warmup_steps: usize,
}

pub struct FinetuneApp {
    batch_size: usize,
    config: String,
    continue_finetune: bool,
    device: Device,
    finetune_embeds: bool,
    max_len: Option<SequenceLength>,
    label_smoothing: Option<f64>,
    mixed_precision: bool,
    summary_writer: Box<dyn SummaryWriter>,
    include_continuations: bool,
    lr_schedule: LrSchedule,
    patience: usize,
    pretrained_model: String,
    saver: BestEpochSaver<f32>,
    train_data: String,
    validation_data: String,
    weight_decay: f64,
}

pub struct LearningRateSchedules {
    pub classifier: PlateauLearningRate<ExponentialDecay>,
    pub encoder: PlateauLearningRate<ExponentialDecay>,
}

impl FinetuneApp {
    pub fn lr_schedules(&self) -> LearningRateSchedules {
        let exp_decay = ExponentialDecay::new(
            self.lr_schedule.initial_lr_classifier.into_inner(),
            self.lr_schedule.lr_decay_rate.into_inner(),
            1,
            false,
            self.lr_schedule.warmup_steps,
        );

        let classifier = PlateauLearningRate::new(
            exp_decay,
            self.lr_schedule.lr_scale.into_inner(),
            self.lr_schedule.lr_patience,
        );

        let mut encoder = classifier.clone();
        encoder.set_initial_lr(self.lr_schedule.initial_lr_encoder.into_inner());

        LearningRateSchedules {
            classifier,
            encoder,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_epoch(
        &self,
        encoders: &Encoders,
        tokenizer: &dyn Tokenize,
        model: &BertModel,
        file: &mut File,
        mut grad_scaler: Option<&mut GradScaler<AdamW>>,
        lr_schedulers: &mut LearningRateSchedules,
        global_step: &mut usize,
        epoch: usize,
    ) -> Result<f32> {
        let epoch_type = if grad_scaler.is_some() {
            "train"
        } else {
            "validation"
        };

        let read_progress = ReadProgress::new(file).context("Cannot create progress bar")?;
        let progress_bar = read_progress.progress_bar().clone();
        progress_bar.set_style(ProgressStyle::default_bar().template(&format!(
            "[Time: {{elapsed_precise}}, ETA: {{eta_precise}}] {{bar}} {{percent}}% {} {{msg}}",
            epoch_type
        )));

        let mut dataset = ConlluDataSet::new(read_progress);

        let mut encoder_accuracy = BTreeMap::new();
        let mut encoder_loss = BTreeMap::new();

        let mut n_tokens = 0;

        // Freeze the encoder during the first epoch.
        let freeze_encoder = epoch == 0;

        for batch in dataset.batches(
            encoders,
            tokenizer,
            self.batch_size,
            self.max_len,
            None,
            true,
        )? {
            let batch = batch?;

            let (lr_classifier, lr_encoder) = if epoch == 0 {
                (self.lr_schedule.initial_lr_classifier.into_inner(), 0.)
            } else {
                (
                    lr_schedulers
                        .classifier
                        .compute_step_learning_rate(*global_step),
                    lr_schedulers
                        .encoder
                        .compute_step_learning_rate(*global_step),
                )
            };

            let attention_mask = seq_len_to_mask(&batch.seq_lens, batch.inputs.size()[1]);

            let n_batch_tokens = i64::from(batch.token_mask.sum(Kind::Int64));

            let model_loss = autocast_or_preserve(self.mixed_precision, || {
                model.loss(
                    &batch.inputs.to_device(self.device),
                    &attention_mask.to_device(self.device),
                    &batch.token_mask.to_device(self.device),
                    &batch
                        .labels
                        .expect("Batch without labels.")
                        .into_iter()
                        .map(|(encoder_name, labels)| (encoder_name, labels.to_device(self.device)))
                        .collect(),
                    self.label_smoothing,
                    grad_scaler.is_some(),
                    FreezeLayers {
                        embeddings: !self.finetune_embeds || freeze_encoder,
                        encoder: freeze_encoder,
                        classifiers: grad_scaler.is_none(),
                    },
                    self.include_continuations,
                )
            });

            n_tokens += n_batch_tokens;

            let scalar_loss: f32 = model_loss.summed_loss.sum(Kind::Float).into();

            if let Some(scaler) = &mut grad_scaler {
                scaler.backward_step(&model_loss.summed_loss.sum(Kind::Float), |name| {
                    let mut config = AdamWConfig::default();

                    // Use weight decay for all variables, except for
                    // layer norm variables.
                    if !name.contains("layer_norm") {
                        config.weight_decay = self.weight_decay;
                    }

                    // Use separate learning rates for the encoder and classifiers.
                    if name.starts_with("classifiers") {
                        config.lr = lr_classifier.into();
                    } else if name.starts_with("encoder") || name.starts_with("embeddings") {
                        config.lr = lr_encoder.into();
                    } else {
                        unreachable!();
                    }

                    config
                });

                if epoch != 0 {
                    self.summary_writer.write_scalar(
                        "gradient_scale",
                        *global_step as i64,
                        scaler.current_scale(),
                    )?;
                }

                if epoch != 0 {
                    *global_step += 1;
                }
            };

            for (encoder_name, loss) in model_loss.encoder_losses {
                *encoder_accuracy.entry(encoder_name.clone()).or_insert(0f32) +=
                    f32::from(&model_loss.encoder_accuracies[&encoder_name])
                        * n_batch_tokens as f32;
                *encoder_loss.entry(encoder_name).or_insert(0f32) +=
                    f32::from(loss) * n_batch_tokens as f32;
            }

            progress_bar.set_message(&format!(
                "classifier lr: {:.1e}, encoder lr: {:.1e}, batch loss: {:.4}, global step: {}",
                lr_classifier, lr_encoder, scalar_loss, global_step
            ));
        }

        progress_bar.finish();

        eprintln!();
        let mut acc_sum = 0.0;
        for (encoder_name, loss) in encoder_loss {
            let acc = encoder_accuracy[&encoder_name] / n_tokens as f32;
            let loss = loss / n_tokens as f32;

            eprintln!("{} loss: {} accuracy: {:.4}", encoder_name, loss, acc);

            self.summary_writer.write_scalar(
                &format!("loss:{},layer:{}", epoch_type, &encoder_name),
                *global_step as i64,
                loss,
            )?;

            self.summary_writer.write_scalar(
                &format!("acc:{},layer:{}", epoch_type, &encoder_name),
                *global_step as i64,
                acc,
            )?;

            acc_sum += acc;
        }
        eprintln!();

        Ok(acc_sum / encoders.len() as f32)
    }
}

impl StickerApp for FinetuneApp {
    fn app() -> App<'static, 'static> {
        let app = App::new("finetune")
            .settings(DEFAULT_CLAP_SETTINGS)
            .about("Finetune a model")
            .arg(
                Arg::with_name(CONFIG)
                    .help("Sticker configuration file")
                    .index(1)
                    .required(true),
            )
            .arg(
                Arg::with_name(CONTINUE)
                    .long("continue")
                    .help("Continue training a sticker model"),
            )
            .arg(
                Arg::with_name(PRETRAINED_MODEL)
                    .help("Pretrained model in HDF5 format")
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
                Arg::with_name(FINETUNE_EMBEDS)
                    .long("finetune-embeds")
                    .help("Finetune embeddings"),
            )
            .arg(
                Arg::with_name(GPU)
                    .long("gpu")
                    .takes_value(true)
                    .help("Use the GPU with the given identifier"),
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
                Arg::with_name(LABEL_SMOOTHING)
                    .long("label-smoothing")
                    .value_name("PROB")
                    .takes_value(true)
                    .help("Distribute the given probability to non-target labels"),
            )
            .arg(
                Arg::with_name(MIXED_PRECISION)
                    .long("mixed-precision")
                    .help("Enable automatic mixed-precision"),
            )
            .arg(
                Arg::with_name(INCLUDE_CONTINUATIONS)
                    .long("include-continuations")
                    .help("Learn to predict continuation label for continuation word pieces"),
            )
            .arg(
                Arg::with_name(MAX_LEN)
                    .long("maxlen")
                    .value_name("N")
                    .takes_value(true)
                    .help("Ignore sentences longer than N tokens"),
            )
            .arg(
                Arg::with_name(LR_DECAY_RATE)
                    .long("lr-decay-rate")
                    .value_name("N")
                    .help("Exponential decay rate")
                    .default_value("0.99998"),
            )
            .arg(
                Arg::with_name(LR_PATIENCE)
                    .long("lr-patience")
                    .value_name("N")
                    .help("Scale learning rate after N epochs without improvement")
                    .default_value("2"),
            )
            .arg(
                Arg::with_name(LR_SCALE)
                    .long("lr-scale")
                    .value_name("SCALE")
                    .help("Value to scale the learning rate by")
                    .default_value("0.9"),
            )
            .arg(
                Arg::with_name(PATIENCE)
                    .long("patience")
                    .value_name("N")
                    .help("Maximum number of epochs without improvement")
                    .default_value("15"),
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
            );

        SummaryOption::add_to_app(app)
    }

    fn parse(matches: &ArgMatches) -> Result<Self> {
        let config = matches.value_of(CONFIG).unwrap().into();
        let pretrained_model = matches
            .value_of(PRETRAINED_MODEL)
            .map(ToOwned::to_owned)
            .unwrap();
        let train_data = matches.value_of(TRAIN_DATA).map(ToOwned::to_owned).unwrap();
        let validation_data = matches
            .value_of(VALIDATION_DATA)
            .map(ToOwned::to_owned)
            .unwrap();
        let batch_size = matches
            .value_of(BATCH_SIZE)
            .unwrap()
            .parse()
            .context("Cannot parse batch size")?;
        let continue_finetune = matches.is_present(CONTINUE);
        let device = match matches.value_of("GPU") {
            Some(gpu) => Device::Cuda(
                gpu.parse()
                    .context(format!("Cannot parse GPU number ({})", gpu))?,
            ),
            None => Device::Cpu,
        };
        let finetune_embeds = matches.is_present(FINETUNE_EMBEDS);
        let initial_lr_classifier = matches
            .value_of(INITIAL_LR_CLASSIFIER)
            .unwrap()
            .parse()
            .context("Cannot parse initial classifier learning rate")?;
        let initial_lr_encoder = matches
            .value_of(INITIAL_LR_ENCODER)
            .unwrap()
            .parse()
            .context("Cannot parse initial encoder learning rate")?;
        let label_smoothing = matches
            .value_of(LABEL_SMOOTHING)
            .map(|v| {
                v.parse()
                    .context(format!("Cannot parse label smoothing probability: {}", v))
            })
            .transpose()?;
        let mixed_precision = matches.is_present(MIXED_PRECISION);
        let summary_writer = SummaryOption::parse(matches)?;
        let max_len = matches
            .value_of(MAX_LEN)
            .map(|v| {
                v.parse()
                    .context(format!("Cannot parse maximum sentence length: {}", v))
            })
            .transpose()?
            .map(SequenceLength::Pieces);
        let include_continuations = matches.is_present(INCLUDE_CONTINUATIONS);
        let lr_decay_rate = matches
            .value_of(LR_DECAY_RATE)
            .unwrap()
            .parse()
            .context("Cannot parse exponential decay rate")?;
        let lr_patience = matches
            .value_of(LR_PATIENCE)
            .unwrap()
            .parse()
            .context("Cannot parse learning rate patience")?;
        let lr_scale = matches
            .value_of(LR_SCALE)
            .unwrap()
            .parse()
            .context("Cannot parse learning rate scale")?;
        let patience = matches
            .value_of(PATIENCE)
            .unwrap()
            .parse()
            .context("Cannot parse patience")?;
        let saver = BestEpochSaver::new("");
        let warmup_steps = matches
            .value_of(WARMUP)
            .unwrap()
            .parse()
            .context("Cannot parse warmup")?;
        let weight_decay = matches
            .value_of(WEIGHT_DECAY)
            .unwrap()
            .parse()
            .context("Cannot parse weight decay")?;

        Ok(FinetuneApp {
            batch_size,
            config,
            continue_finetune,
            device,
            finetune_embeds,
            max_len,
            label_smoothing,
            mixed_precision,
            include_continuations,
            summary_writer,
            lr_schedule: LrSchedule {
                initial_lr_classifier,
                initial_lr_encoder,
                lr_decay_rate,
                lr_patience,
                lr_scale,
                warmup_steps,
            },
            patience,
            pretrained_model,
            saver,
            train_data,
            validation_data,
            weight_decay,
        })
    }

    fn run(&self) -> Result<()> {
        let model = if self.continue_finetune {
            Model::load_from(&self.config, &self.pretrained_model, self.device, false)?
        } else {
            Model::load_from_hdf5(&self.config, &self.pretrained_model, self.device)?
        };

        let mut train_file = File::open(&self.train_data)
            .context(format!("Cannot open train data file: {}", self.train_data))?;
        let mut validation_file = File::open(&self.validation_data).context(format!(
            "Cannot open validation data file: {}",
            self.validation_data
        ))?;

        let mut saver = self.saver.clone();
        let opt = AdamW::new(&model.vs);
        let mut grad_scaler = GradScaler::new_with_defaults(self.mixed_precision, opt);

        let mut lr_schedules = self.lr_schedules();

        let mut last_acc = 0.0;
        let mut best_acc = 0.0;
        let mut best_epoch = 0;

        let mut global_step = 1;

        for epoch in 0.. {
            eprintln!("Epoch {}", epoch);

            let _ = lr_schedules
                .classifier
                .compute_epoch_learning_rate(epoch, last_acc);
            let _ = lr_schedules
                .encoder
                .compute_epoch_learning_rate(epoch, last_acc);

            self.run_epoch(
                &model.encoders,
                &*model.tokenizer,
                &model.model,
                &mut train_file,
                Some(&mut grad_scaler),
                &mut lr_schedules,
                &mut global_step,
                epoch,
            )
            .context("Cannot run train epoch")?;

            last_acc = self
                .run_epoch(
                    &model.encoders,
                    &*model.tokenizer,
                    &model.model,
                    &mut validation_file,
                    None,
                    &mut lr_schedules,
                    &mut global_step,
                    epoch,
                )
                .context("Cannot run valdidation epoch")?;

            if last_acc > best_acc {
                best_epoch = epoch;
                best_acc = last_acc;
            }

            saver
                .save(&model.vs, CompletedUnit::Epoch(last_acc))
                .context("Error saving model")?;

            let epoch_status = if best_epoch == epoch { "🎉" } else { "" };
            eprintln!(
                "Epoch {} (validation): acc: {:.4}, best epoch: {}, best acc: {:.4} {}",
                epoch, last_acc, best_epoch, best_acc, epoch_status
            );

            self.summary_writer
                .write_scalar("acc:validation,avg", global_step as i64, last_acc)?;

            if epoch - best_epoch == self.patience {
                eprintln!(
                    "Lost my patience! Best epoch: {} with accuracy: {:.4}",
                    best_epoch, best_acc
                );
                break;
            }
        }

        Ok(())
    }
}
