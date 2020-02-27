use std::borrow::Borrow;

use tch::nn::{Init, Path};
use tch::Tensor;

#[derive(Clone, Copy, Debug)]
pub struct BilinearConfig {
    pub features: i64,
    pub initializer_range: f64,
}

#[derive(Debug)]
pub struct Bilinear {
    weight: Tensor,
}

impl Bilinear {
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: &BilinearConfig) -> Self {
        let vs = vs.borrow();

        let weight = vs.var(
            "weight",
            &[config.features, config.features],
            Init::Randn {
                mean: 0.,
                stdev: config.initializer_range,
            },
        );

        Bilinear { weight }
    }

    pub fn forward(&self, input1: &Tensor, input2: &Tensor) -> Tensor {
        assert_eq!(
            input1.size(),
            input2.size(),
            "Inputs to Bilinear must have the same shapes: {:?} {:?}",
            input1.size(),
            input2.size()
        );

        // The shapes of the inputs are [batch_size, max_seq_len, features].
        // After matrix multiplication, we the intermediate shape
        // [batch_size, max_seq_len, features].
        let intermediate = input1.matmul(&self.weight);

        // Transpose the second input to obtain the shape
        // [batch_size, features, max_seq_len]. The perform a matrix
        // multiplication to get the output with the shape
        // [batch_size, seq_len, seq_len].
        intermediate.matmul(&input2.transpose(1, 2))
    }
}

#[cfg(test)]
mod tests {
    use tch::nn::VarStore;
    use tch::{Device, Kind, Tensor};

    use crate::model::bilinear::{Bilinear, BilinearConfig};

    #[test]
    fn bilinear_correct_shapes() {
        // Apply a bilinear layer to ensure that the shapes are correct.

        let input1 = Tensor::rand(&[64, 10, 200], (Kind::Float, Device::Cpu));
        let input2 = Tensor::rand(&[64, 10, 200], (Kind::Float, Device::Cpu));

        let vs = VarStore::new(Device::Cpu);
        let bilinear = Bilinear::new(
            vs.root(),
            &BilinearConfig {
                features: 200,
                initializer_range: 0.02,
            },
        );

        assert_eq!(bilinear.forward(&input1, &input2).size(), &[64, 10, 10]);
    }
}
