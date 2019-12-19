//! Encoder configuration and construction.

mod config;
pub use config::{DependencyEncoder, EncoderType, EncodersConfig};

#[allow(clippy::module_inception)]
mod encoders;
pub use encoders::{Encoder, Encoders};
