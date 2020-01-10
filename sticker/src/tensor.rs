use std::collections::HashMap;
use std::convert::TryInto;

use ndarray::{s, Array1, Array2, ArrayView1};
use tch::Tensor;

mod labels {
    pub trait Labels {
        fn from_shape(
            encoder_names: impl IntoIterator<Item = impl Into<String>>,
            batch_size: usize,
            time_steps: usize,
        ) -> Self;
    }
}

/// No labels.
#[derive(Default)]
pub struct NoLabels;

impl labels::Labels for NoLabels {
    fn from_shape(
        _encoder_names: impl IntoIterator<Item = impl Into<String>>,
        _batch_size: usize,
        _time_steps: usize,
    ) -> Self {
        NoLabels
    }
}

/// Labels per encoder.
pub struct LabelTensor {
    labels: HashMap<String, Array2<i64>>,
    label_mask: Array2<i32>,
}

impl labels::Labels for LabelTensor {
    fn from_shape(
        encoder_names: impl IntoIterator<Item = impl Into<String>>,
        batch_size: usize,
        time_steps: usize,
    ) -> Self {
        let labels = encoder_names
            .into_iter()
            .map(Into::into)
            .map(|encoder_name| (encoder_name, Array2::zeros((batch_size, time_steps))))
            .collect();

        LabelTensor {
            labels,
            label_mask: Array2::zeros((batch_size, time_steps)),
        }
    }
}

/// Build Torch `Tensor`s from `ndarray` vectors.
pub struct TensorBuilder<L> {
    current_sequence: usize,
    inputs: Array2<i64>,
    labels: L,
    seq_lens: Array1<i32>,
}

impl<L> TensorBuilder<L>
where
    L: labels::Labels,
{
    /// Create a new `TensorBuilder`.
    ///
    /// Creates a new builder with the given batch size, number of time steps,
    /// and encoder names.
    pub fn new(
        batch_size: usize,
        max_seq_len: usize,
        encoder_names: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        let labels = L::from_shape(encoder_names, batch_size, max_seq_len);
        TensorBuilder {
            current_sequence: 0,
            inputs: Array2::zeros((batch_size, max_seq_len)),
            labels,
            seq_lens: Array1::zeros((batch_size,)),
        }
    }

    /// Add an instance without labels.
    pub fn add_without_labels(&mut self, input: ArrayView1<i64>) {
        assert!(
            self.current_sequence < self.inputs.shape()[0],
            "TensorBuilder is already filled."
        );

        #[allow(clippy::deref_addrof)]
        self.inputs
            .row_mut(self.current_sequence)
            .slice_mut(s![0..input.len()])
            .assign(&input);

        self.seq_lens[self.current_sequence] = input.len() as i32;

        self.current_sequence += 1
    }
}

impl TensorBuilder<LabelTensor> {
    /// Add an instance with labels.
    pub fn add_with_labels(
        &mut self,
        input: ArrayView1<i64>,
        labels: HashMap<&str, Array1<i64>>,
        label_mask: ArrayView1<i32>,
    ) {
        assert!(
            self.current_sequence < self.inputs.shape()[0],
            "TensorBuilder is already filled."
        );

        assert_eq!(
            self.labels.labels.len(),
            labels.len(),
            "Expected labels for {} encoders, got labels for {}",
            self.labels.labels.len(),
            labels.len(),
        );

        #[allow(clippy::deref_addrof)]
        self.labels
            .label_mask
            .row_mut(self.current_sequence)
            .slice_mut(s![0..label_mask.len()])
            .assign(&label_mask);

        for (encoder_name, labels) in labels {
            assert_eq!(
                labels.len(),
                label_mask.len(),
                "Input for encoder {} has length {}, but the mask length is {}",
                encoder_name,
                labels.len(),
                label_mask.len()
            );

            #[allow(clippy::deref_addrof)]
            self.labels
                .labels
                .get_mut(encoder_name)
                .unwrap_or_else(|| panic!("Undefined encoder: {}", encoder_name))
                .row_mut(self.current_sequence)
                .slice_mut(s![0..input.len()])
                .assign(&labels)
        }

        self.add_without_labels(input);
    }
}

/// Tensors constructed by `TensorBuilder`.
#[derive(Debug)]
pub struct Tensors {
    /// Input representations.
    pub inputs: Tensor,

    /// Labels.
    pub labels: Option<HashMap<String, Tensor>>,

    /// Label mask.
    pub label_mask: Option<Tensor>,

    /// Sequence lengths.
    pub seq_lens: Tensor,
}

impl From<TensorBuilder<NoLabels>> for Tensors {
    fn from(builder: TensorBuilder<NoLabels>) -> Self {
        Tensors {
            inputs: builder.inputs.try_into().unwrap(),
            labels: None,
            label_mask: None,
            seq_lens: builder.seq_lens.try_into().unwrap(),
        }
    }
}

impl From<TensorBuilder<LabelTensor>> for Tensors {
    fn from(builder: TensorBuilder<LabelTensor>) -> Self {
        let labels = builder
            .labels
            .labels
            .into_iter()
            .map(|(encoder_name, matrix)| (encoder_name, matrix.try_into().unwrap()))
            .collect();

        Tensors {
            inputs: builder.inputs.try_into().unwrap(),
            labels: Some(labels),
            label_mask: Some(builder.labels.label_mask.try_into().unwrap()),
            seq_lens: builder.seq_lens.try_into().unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;
    use tch::Tensor;

    use super::{LabelTensor, NoLabels, TensorBuilder, Tensors};

    #[test]
    fn instances_are_added() {
        let mut builder: TensorBuilder<NoLabels> = TensorBuilder::new(2, 3, vec!["a", "b"]);
        builder.add_without_labels(arr1(&[1, 2]).view());
        builder.add_without_labels(arr1(&[3, 4, 5]).view());

        let tensors: Tensors = builder.into();

        // No labels.
        assert_eq!(tensors.labels, None);
        assert_eq!(tensors.label_mask, None);

        assert_eq!(tensors.seq_lens, Tensor::of_slice(&[2, 3]));
        assert_eq!(
            tensors.inputs,
            Tensor::of_slice(&[1, 2, 0, 3, 4, 5]).reshape(&[2, 3])
        );
    }

    #[test]
    fn instances_are_added_with_labels() {
        let mut builder: TensorBuilder<LabelTensor> = TensorBuilder::new(2, 3, vec!["a", "b"]);
        builder.add_with_labels(
            arr1(&[1, 2]).view(),
            vec![("a", arr1(&[11, 12])), ("b", arr1(&[21, 22]))]
                .into_iter()
                .collect(),
            arr1(&[1, 0]).view(),
        );
        builder.add_with_labels(
            arr1(&[3, 4, 5]).view(),
            vec![("a", arr1(&[13, 14, 15])), ("b", arr1(&[23, 24, 25]))]
                .into_iter()
                .collect(),
            arr1(&[1, 0, 1]).view(),
        );

        let tensors: Tensors = builder.into();

        // Labels.
        assert_eq!(
            tensors.labels,
            Some(
                vec![
                    (
                        "a".to_string(),
                        Tensor::of_slice(&[11, 12, 0, 13, 14, 15]).reshape(&[2, 3])
                    ),
                    (
                        "b".to_string(),
                        Tensor::of_slice(&[21, 22, 0, 23, 24, 25]).reshape(&[2, 3])
                    )
                ]
                .into_iter()
                .collect()
            )
        );
        assert_eq!(
            tensors.label_mask,
            Some(Tensor::of_slice(&[1, 0, 0, 1, 0, 1]).reshape(&[2, 3]))
        );

        assert_eq!(tensors.seq_lens, Tensor::of_slice(&[2, 3]));
        assert_eq!(
            tensors.inputs,
            Tensor::of_slice(&[1, 2, 0, 3, 4, 5]).reshape(&[2, 3])
        );
    }

    #[should_panic]
    #[test]
    fn panics_when_labels_and_mask_len_differ() {
        let mut builder: TensorBuilder<LabelTensor> = TensorBuilder::new(2, 3, vec!["a", "b"]);
        builder.add_with_labels(
            arr1(&[1, 2]).view(),
            vec![("a", arr1(&[11])), ("b", arr1(&[21, 22]))]
                .into_iter()
                .collect(),
            arr1(&[1, 0]).view(),
        );
    }

    #[should_panic]
    #[test]
    fn panics_when_too_many_instances_pushed() {
        let mut builder: TensorBuilder<NoLabels> = TensorBuilder::new(1, 3, vec!["a", "b"]);
        builder.add_without_labels(arr1(&[1, 2]).view());
        builder.add_without_labels(arr1(&[3, 4, 5]).view());
    }

    #[should_panic]
    #[test]
    fn panics_when_labels_for_encoder_missing() {
        let mut builder: TensorBuilder<LabelTensor> = TensorBuilder::new(2, 3, vec!["a", "b"]);
        builder.add_with_labels(
            arr1(&[1, 2]).view(),
            vec![("b", arr1(&[21, 22]))].into_iter().collect(),
            arr1(&[1, 0]).view(),
        );
    }
}
