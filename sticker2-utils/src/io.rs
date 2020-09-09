use std::cell::RefCell;
use std::fs::File;
use std::io::BufWriter;

use anyhow::{Context, Result};
use sticker2::config::{Config, PretrainConfig, TomlRead};
use sticker2::encoders::Encoders;
use sticker2::input::Tokenize;
use sticker2::model::bert::BertModel;
use tch::nn::VarStore;
use tch::Device;
use tfrecord::{EventWriter, EventWriterInit};

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
    pub fn load(config_path: &str, device: Device, freeze: bool) -> Result<Self> {
        let config = load_config(config_path)?;
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
    ) -> Result<Model> {
        let config = load_config(config_path)?;
        let encoders = load_encoders(&config)?;
        let tokenizer = load_tokenizer(&config)?;
        let pretrain_config = load_pretrain_config(&config)?;

        let mut vs = VarStore::new(device);

        let model = BertModel::new(
            vs.root(),
            &pretrain_config,
            &encoders,
            0.0,
            config.model.position_embeddings,
        )
        .context("Cannot construct model")?;

        vs.load(parameters_path)
            .context("Cannot load model parameters")?;

        if freeze {
            vs.freeze();
        }

        Ok(Model {
            encoders,
            model,
            tokenizer,
            vs,
        })
    }

    /// Load a model on the given device.
    ///
    /// In contrast to `load_model`, this does not load the parameters
    /// specified in the configuration file, but the parameters from
    /// the HDF5 file at `hdf5_path`.
    #[cfg(feature = "load-hdf5")]
    pub fn load_from_hdf5(config_path: &str, hdf5_path: &str, device: Device) -> Result<Model> {
        let config = load_config(config_path)?;
        let encoders = load_encoders(&config)?;
        let tokenizer = load_tokenizer(&config)?;
        let pretrain_config = load_pretrain_config(&config)?;

        let vs = VarStore::new(device);

        let model =
            BertModel::from_pretrained(vs.root(), &pretrain_config, hdf5_path, &encoders, 0.5)
                .context("Cannot load pretrained model parameters")?;

        Ok(Model {
            encoders,
            model,
            tokenizer,
            vs,
        })
    }

    #[cfg(not(feature = "load-hdf5"))]
    pub fn load_from_hdf5(_config_path: &str, _hdf5_path: &str, _device: Device) -> Result<Model> {
        anyhow::bail!("Cannot load HDF5 model: sticker2 was compiled without support for HDF5");
    }
}

pub fn load_pretrain_config(config: &Config) -> Result<PretrainConfig> {
    config
        .model
        .pretrain_config()
        .context("Cannot load pretraining model configuration")
}

pub fn load_config(config_path: &str) -> Result<Config> {
    let config_file = File::open(config_path)
        .context(format!("Cannot open configuration file '{}'", &config_path))?;
    let mut config = Config::from_toml_read(config_file)
        .context(format!("Cannot parse configuration file: {}", config_path))?;
    config.relativize_paths(config_path).context(format!(
        "Cannot relativize paths in configuration file: {}",
        config_path
    ))?;

    Ok(config)
}

fn load_encoders(config: &Config) -> Result<Encoders> {
    let f = File::open(&config.labeler.labels)
        .context(format!("Cannot open label file: {}", config.labeler.labels))?;
    let encoders: Encoders = serde_yaml::from_reader(&f).context(format!(
        "Cannot deserialize labels from: {}",
        config.labeler.labels
    ))?;

    for encoder in &*encoders {
        eprintln!(
            "Loaded labels for encoder '{}': {} labels",
            encoder.name(),
            encoder.encoder().len()
        );
    }

    Ok(encoders)
}

pub fn load_tokenizer(config: &Config) -> Result<Box<dyn Tokenize>> {
    config
        .tokenizer()
        .context("Cannot read tokenizer vocabulary")
}

pub struct TensorBoardWriter {
    writer: Option<RefCell<EventWriter<BufWriter<File>>>>,
}

impl TensorBoardWriter {
    pub fn new(prefix: impl AsRef<str>) -> Result<Self> {
        let writer = EventWriterInit::default().from_prefix(prefix, None)?;
        Ok(TensorBoardWriter {
            writer: Some(RefCell::new(writer)),
        })
    }

    pub fn noop() -> Self {
        TensorBoardWriter { writer: None }
    }

    pub fn write_scalar(&self, tag: &str, step: i64, value: f32) -> Result<()> {
        if let Some(ref writer) = self.writer {
            writer.borrow_mut().write_scalar(tag, step, value)?;
        }

        Ok(())
    }
}
