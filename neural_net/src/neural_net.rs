use crate::{
    layers::Layer,
    loss::LossFunction,
    saving_and_loading::{Format, save_to_file},
};
use anyhow::Result;
use ndarray::{Array2, ArrayView, ArrayView2, Ix2};
use rand::{Rng, prelude::SliceRandom};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{fmt::Display, path::Path};

#[derive(Serialize, Deserialize, Debug)]
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
}

#[derive(Serialize, Deserialize, Debug)]
struct TrainingData {
    nabla_w: Vec<Array2<f64>>,
    nabla_b: Vec<Array2<f64>>,
    delta_nabla_w: Vec<Array2<f64>>,
    delta_nabla_b: Vec<Array2<f64>>,
    activations: Vec<Array2<f64>>,
    pre_activations: Vec<Array2<f64>>,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>) -> Self {
        NeuralNetwork { layers }
    }

    pub fn feedforward(&self, inputs: ArrayView2<f64>) -> Array2<f64> {
        let mut result = inputs.to_owned();
        for layer in self.layers.iter() {
            result = layer.weights.dot(&result);
            result += &layer.biases;
            result.mapv_inplace(|v| layer.activation.apply(v));
        }
        result
    }

    pub fn validate(
        &self,
        data: &[(ArrayView2<f64>, ArrayView2<f64>)],
        loss: &LossFunction,
    ) -> f64 {
        let (sum, count) = data
            .par_iter()
            .map(|(input, truth)| {
                let output = self.feedforward(*input);
                loss.apply(output.view(), *truth)
            })
            .fold(|| (0.0f64, 0usize), |(acc, cnt), l| (acc + l, cnt + 1))
            .reduce(|| (0.0f64, 0usize), |(a1, c1), (a2, c2)| (a1 + a2, c1 + c2));
        if count == 0 {
            0.0
        } else {
            sum / (count as f64)
        }
    }

    pub fn train(
        &mut self,
        raw_training_data: Vec<(Array2<f64>, Array2<f64>)>,
        epochs: usize,
        batches_per_epoch: usize,
        batch_size: usize,
        learning_rate: f64,
        lambda: f64,
        rng: &mut impl Rng,
        loss_function: LossFunction,
    ) {
        let weight_size: Vec<Array2<f64>> = self
            .layers
            .iter()
            .map(|w| Array2::zeros(w.weights.dim()))
            .collect();
        let bias_size: Vec<Array2<f64>> = self
            .layers
            .iter()
            .map(|w| Array2::zeros(w.biases.dim()))
            .collect();
        let mut activations_shape: Vec<Array2<f64>> = weight_size
            .iter()
            .map(|w| Array2::zeros((w.dim().1, 1)))
            .collect();

        activations_shape.push(Array2::zeros((
            self.layers.last().unwrap().biases.dim().0,
            1,
        )));

        let mut training_data = TrainingData {
            nabla_w: weight_size.clone(),
            nabla_b: bias_size.clone(),
            delta_nabla_w: weight_size,
            delta_nabla_b: bias_size,
            activations: activations_shape.clone(),
            pre_activations: activations_shape,
        };

        let mut batches = raw_training_data;
        batches.shuffle(rng);

        #[cfg(feature = "loss")]
        let validation_pairs = {
            let val_size = ((batches.len() as f64 * 0.1).ceil() as usize).clamp(1, 10);
            batches.drain(..val_size).collect::<Vec<_>>()
        };
        #[cfg(feature = "loss")]
        let views = validation_pairs
            .iter()
            .map(|(i, o)| (i.view(), o.view()))
            .collect::<Vec<(ArrayView2<f64>, ArrayView2<f64>)>>();

        for epoch in 0..epochs {
            batches.shuffle(rng);

            let start = std::time::Instant::now();

            for batch in batches.chunks(batch_size).cycle().take(batches_per_epoch) {
                self.update_weights_biases(
                    batch,
                    learning_rate,
                    lambda,
                    &mut training_data,
                    &loss_function,
                );
            }

            let elapsed = start.elapsed();

            #[cfg(not(feature = "loss"))]
            {
                println!("Epoch {epoch} - Time: {elapsed:?}");
            }

            #[cfg(feature = "loss")]
            {
                let loss = self.validate(&views, &loss_function);
                println!(
                    "Epoch {epoch} - Loss: {loss} - Time: {elapsed:?} | {:?}",
                    start.elapsed() - elapsed
                );
                if loss.is_nan() {
                    panic!("Loss is NaN. Training aborted.");
                }
            }
        }
    }

    fn update_weights_biases(
        &mut self,
        batch: &[(Array2<f64>, Array2<f64>)],
        learning_rate: f64,
        lambda: f64,
        training_data: &mut TrainingData,
        loss: &LossFunction,
    ) {
        let nabla_w = &mut training_data.nabla_w;
        let nabla_b = &mut training_data.nabla_b;
        let delta_nabla_w = &mut training_data.delta_nabla_w;
        let delta_nabla_b = &mut training_data.delta_nabla_b;
        let activations = &mut training_data.activations;
        let pre_activations = &mut training_data.pre_activations;
        reset_matrix(nabla_w);
        reset_matrix(nabla_b);
        reset_matrix(delta_nabla_w);
        reset_matrix(delta_nabla_b);
        for (i, t) in batch {
            self.backpropagation(
                i,
                t,
                delta_nabla_w,
                delta_nabla_b,
                activations,
                pre_activations,
                loss,
            );
            for (nw, dnw) in nabla_w.iter_mut().zip(delta_nabla_w.iter()) {
                *nw += dnw;
            }
            for (nb, dnb) in nabla_b.iter_mut().zip(delta_nabla_b.iter()) {
                *nb += dnb;
            }
            reset_matrix(delta_nabla_w);
            reset_matrix(delta_nabla_b);
        }

        let batch_size_f = batch.len() as f64;
        for (layer, grad_w) in self.layers.iter_mut().zip(nabla_w.iter()) {
            if lambda != 0.0 {
                // decouple decay: w := w * (1 - lr * lambda / batch_size)
                layer
                    .weights
                    .mapv_inplace(|w| w * (1.0 - learning_rate * lambda / batch_size_f));
            }
            layer
                .weights
                .scaled_add(-learning_rate / batch_size_f, grad_w);
        }

        for (layer, grad_b) in self.layers.iter_mut().zip(nabla_b.iter()) {
            layer
                .biases
                .scaled_add(-learning_rate / batch_size_f, grad_b);
        }
    }

    fn backpropagation(
        &mut self,
        input: &Array2<f64>,
        truth: &Array2<f64>,
        delta_nabla_w: &mut [Array2<f64>],
        delta_nabla_b: &mut [Array2<f64>],
        activations: &mut [Array2<f64>],
        pre_activations: &mut [Array2<f64>],
        loss: &LossFunction,
    ) {
        let num_layers = self.layers.len();

        activations[0].assign(input);
        for (i, layer) in self.layers.iter().enumerate() {
            let mut z = layer.weights.dot(&activations[i]);
            z += &layer.biases;
            pre_activations[i + 1].assign(&z);

            let mut a = z.clone();
            a.mapv_inplace(|v| layer.activation.apply(v));
            activations[i + 1].assign(&a);
        }

        let a_L = activations[num_layers].view();
        let z_L = pre_activations[num_layers].view();
        let mut delta = loss.derivative(a_L, truth.view()); // dL/da

        if matches!(loss, LossFunction::MeanErrorSquared) {
            let output_activation = &self.layers[num_layers - 1].activation;
            delta.zip_mut_with(&z_L, |d, &z| {
                *d *= output_activation.derivative(z);
            });
        }

        delta_nabla_b[num_layers - 1].assign(&delta);
        delta_nabla_w[num_layers - 1].assign(&delta.dot(&activations[num_layers - 1].t()));

        for l in (0..num_layers - 1).rev() {
            // backpropagate through weights
            delta = self.layers[l + 1].weights.t().dot(&delta);

            let z_l = pre_activations[l + 1].view();
            let activation_l = &self.layers[l].activation;
            delta.zip_mut_with(&z_l, |d, &z| {
                *d *= activation_l.derivative(z);
            });

            delta_nabla_b[l].assign(&delta);
            delta_nabla_w[l].assign(&delta.dot(&activations[l].t()));
        }
    }

    fn save_checkpoint(&self, epoch: usize) -> Result<()> {
        save_to_file(&format!("checkpoint_{epoch}.bin"), self, Format::Binary)
    }
}

pub fn print_matrix<T: Display>(matrix: &ArrayView<T, Ix2>) {
    for row in matrix.rows() {
        for cell in row {
            print!("{cell:.4}, ");
        }
        println!();
    }
}

#[inline(always)]
pub fn reset_matrix<O: num_traits::Zero + Copy>(i: &mut [Array2<O>]) {
    for x in i.iter_mut() {
        x.fill(O::zero());
    }
}
