use crate::layers::Layer;
use log::info;
use ndarray::{Array2, ArrayBase, ArrayView, ArrayView2, Ix2, OwnedRepr};
use rand::{prelude::SliceRandom, Rng};
use rayon::prelude::*;
use std::fmt::Display;

pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
}

struct TrainingData {
    nabla_w: Vec<Array2<f64>>,
    nabla_b: Vec<Array2<f64>>,
    delta_nabla_w: Vec<Array2<f64>>,
    delta_nabla_b: Vec<Array2<f64>>,
    activations: Vec<Array2<f64>>,
    pre_activations: Vec<Array2<f64>>,
}

fn cost_derivative(output: ArrayView2<f64>, truth: ArrayView2<f64>) -> Array2<f64> {
    &output - &truth
}

fn error_squared(output: ArrayView2<f64>, truth: ArrayView2<f64>) -> f64 {
    let diff = &output - &truth;
    diff.mapv(|d| d.powi(2)).sum() / (output.len() as f64)
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

    pub fn validate<'a>(&self, data: &[(ArrayView2<'a, f64>, ArrayView2<'a, f64>)]) -> f64 {
        let (sum, count) = data
            .par_iter()
            .map(|(input, truth)| {
                let output = self.feedforward(*input);
                error_squared(output.view(), *truth)
            })
            .fold(
                || (0.0f64, 0usize),
                |(acc, cnt), loss| (acc + loss, cnt + 1),
            )
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

        #[cfg(feature = "logging")]
        let validation_pairs = {
            let val_size = 10.min(batch_size / 10);
            batches.drain(..val_size).collect::<Vec<_>>()
        };
        #[cfg(feature = "logging")]
        let views = validation_pairs
            .iter()
            .map(|(i, o)| (i.view(), o.view()))
            .collect::<Vec<(ArrayView2<f64>, ArrayView2<f64>)>>();

        for epoch in 0..epochs {
            batches.shuffle(rng);

            let start = std::time::Instant::now();

            for batch in batches.chunks(batch_size).cycle().take(batches_per_epoch) {
                self.update_weights_biases(batch, learning_rate, lambda, &mut training_data);
            }

            #[cfg(feature = "logging")]
            {
                let elapsed = start.elapsed();
                let loss = self.validate(&views);
                println!(
                    "Epoch {epoch} - Loss: {loss} - Time: {elapsed:?}|{:?}",
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
        for (o, t) in batch {
            self.backpropagation(
                o,
                t,
                delta_nabla_w,
                delta_nabla_b,
                activations,
                pre_activations,
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
            let mut adjusted = grad_w.clone();
            if lambda != 0.0 {
                adjusted += &layer.weights.mapv(|w| lambda * w);
            }
            layer
                .weights
                .scaled_add(-(learning_rate / batch_size_f), &adjusted);
        }

        for (b, nb) in self.layers.iter_mut().zip(nabla_b.iter()) {
            b.biases
                .scaled_add(-(learning_rate / batch.len() as f64), nb);
        }
    }

    fn backpropagation(
        &mut self,
        input: &Array2<f64>,
        truth: &Array2<f64>,
        delta_nabla_w: &mut [Array2<f64>],
        delta_nabla_b: &mut [Array2<f64>],
        activations: &mut [Array2<f64>],
        post_biases: &mut [Array2<f64>],
    ) {
        let mut previous_output: Array2<f64> = input.clone(); // output from the previous layer
        activations[0].assign(&previous_output);
        // feedforward
        let mut z: Array2<f64>;
        for (i, layer) in self.layers.iter().enumerate() {
            z = layer.weights.dot(&previous_output);
            z += &layer.biases;
            post_biases[i + 1].assign(&z);
            z.mapv_inplace(|v| layer.activation.apply(v));
            previous_output = z;
            activations[i + 1].assign(&previous_output);
        }

        // backward pass
        let last_activation = activations.last().unwrap();
        let mut delta: Array2<f64> = cost_derivative(last_activation.view(), truth.view());
        delta.zip_mut_with(post_biases.last().unwrap(), |d, s| {
            *d *= self.layers.last().unwrap().activation.derivative(*s)
        });
        let nabla_b_len = delta_nabla_b.len();
        delta_nabla_b[nabla_b_len - 1].assign(&delta);
        let nabla_w_len = delta_nabla_w.len();

        delta_nabla_w[nabla_w_len - 1].assign(&delta.dot(&activations[activations.len() - 2].t()));

        for layer_idx in (1..self.layers.len()).rev() {
            // pre-activation of current layer is post_biases[layer_idx]
            let pre_act = &post_biases[layer_idx];
            let prev_activation = &activations[layer_idx - 1];

            let mut delta_derivative = pre_act.clone();
            delta_derivative.mapv_inplace(|v| self.layers[layer_idx - 1].activation.derivative(v));

            delta = self.layers[layer_idx].weights.t().dot(&delta);
            delta.zip_mut_with(&delta_derivative, |d, s| *d *= s);

            delta_nabla_b[layer_idx - 1].assign(&delta);
            delta_nabla_w[layer_idx - 1].assign(&delta.dot(&prev_activation.t()));
        }
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

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    let diff = (a - b).abs();
    diff <= tol.max(tol * b.abs().max(a.abs()))
}

fn array_approx_eq(a: &Array2<f64>, b: &Array2<f64>, tol: f64) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y, tol))
}
