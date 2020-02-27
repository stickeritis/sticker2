use std::borrow::Borrow;

use tch::nn::{Init, Path};
use tch::Tensor;

/// Configuration for the `Bilinear` layer.
#[derive(Clone, Copy, Debug)]
pub struct PairwiseBilinearConfig {
    /// The number of features.
    pub features: i64,

    /// Standard deviation for random initialization.
    pub initializer_range: f64,
}

/// Pairwise bilinear forms.
///
/// Given two batches with sequence length *n*, apply pairwise
/// bilinear forms to each timestep within a sequence.
#[derive(Debug)]
pub struct PairwiseBilinear {
    weight: Tensor,
}

impl PairwiseBilinear {
    /// Construct a new bilinear layer.
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: &PairwiseBilinearConfig) -> Self {
        let vs = vs.borrow();

        let weight = vs.var(
            "weight",
            &[config.features, config.features],
            Init::Randn {
                mean: 0.,
                stdev: config.initializer_range,
            },
        );

        PairwiseBilinear { weight }
    }

    /// Apply this layer to the given inputs.
    ///
    /// Both inputs must have the same shape. Returns a tensor of
    /// shape `[batch_size, seq_len, seq_len]` given inputs of shape
    /// `[batch_size, seq_len, features]`.
    pub fn forward(&self, input1: &Tensor, input2: &Tensor) -> Tensor {
        assert_eq!(
            input1.size(),
            input2.size(),
            "Inputs to Bilinear must have the same shape: {:?} {:?}",
            input1.size(),
            input2.size()
        );

        assert_eq!(
            input1.dim(),
            3,
            "Shape should have 3 dimensions, has: {}",
            input1.dim()
        );

        // The shapes of the inputs are [batch_size, max_seq_len, features].
        // After matrix multiplication, we get the intermediate shape
        // [batch_size, max_seq_len, features].
        let intermediate = input1.matmul(&self.weight);

        // Transpose the second input to obtain the shape
        // [batch_size, features, max_seq_len]. We perform a matrix
        // multiplication to get the output with the shape
        // [batch_size, seq_len, seq_len].
        intermediate.matmul(&input2.transpose(1, 2))
    }
}

#[cfg(test)]
mod tests {
    use tch::nn::VarStore;
    use tch::{Device, Kind, Tensor};

    use crate::model::pairwise_bilinear::{PairwiseBilinear, PairwiseBilinearConfig};

    #[test]
    fn bilinear_correct_shapes() {
        // Apply a bilinear layer to ensure that the shapes are correct.

        let input1 = Tensor::rand(&[64, 10, 200], (Kind::Float, Device::Cpu));
        let input2 = Tensor::rand(&[64, 10, 200], (Kind::Float, Device::Cpu));

        let vs = VarStore::new(Device::Cpu);
        let bilinear = PairwiseBilinear::new(
            vs.root(),
            &PairwiseBilinearConfig {
                features: 200,
                initializer_range: 0.02,
            },
        );

        assert_eq!(bilinear.forward(&input1, &input2).size(), &[64, 10, 10]);
    }
}
