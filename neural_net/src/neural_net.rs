use crate::layers::Layer;
use ndarray::{Array2, ArrayBase, ArrayView, Ix2, OwnedRepr};
use rand::prelude::SliceRandom;
use rand::Rng;
use std::fmt::Display;

pub struct NeuralNetwork {
    pub layers: Vec<Layer>
}

struct TrainingData {
    nabla_w: Vec<Array2<f64>>,
    nabla_b: Vec<Array2<f64>>,
    delta_nabla_w: Vec<Array2<f64>>,
    delta_nabla_b: Vec<Array2<f64>>,
    activations: Vec<Array2<f64>>,
    post_biases: Vec<Array2<f64>>
}

fn cost_derivative(output: &ArrayBase<OwnedRepr<f64>, Ix2>, truth: &Array2<f64>) -> Array2<f64> {
    output - truth
}

fn error_squared(output: &Array2<f64>, truth: &Array2<f64>) -> f64 {
    output
        .iter()
        .zip(truth.iter())
        .map(|(o, t)| (o - t).powi(2)).sum::<f64>()
        / (output.len() as f64)
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>) -> Self {
        NeuralNetwork { layers }
    }

    //fn initialise_random(layer_sizes: Vec<usize>) -> Self {
    //    let weights  = (1..layer_sizes.len()).map(|i| gen_random_matrix(layer_sizes[i], layer_sizes[i - 1])).collect();
    //    let biases = (1..layer_sizes.len()).map(|i| gen_random_matrix(layer_sizes[i], 1)).collect();
    //    NeuralNetwork::new(weights, biases)
    //}

    pub fn feedforward(&self, inputs: &Array2<f64>) -> Array2<f64> {
        let mut result = inputs.to_owned();
        for layer in self.layers.iter() {
            result = layer.weights.dot(&result);
            //dbg!("dot", &result, &layer.biases);
            result += &layer.biases;
            result.mapv_inplace(layer.activation_function);
        }
        result
    }

    pub fn validate(&self, inputs: &[Array2<f64>], truths: &[Array2<f64>]) -> f64 {
        inputs
            .iter()
            .zip(truths.iter())
            .map(|(i, o)| error_squared(&self.feedforward(i), o))
            .sum::<f64>()
            / (inputs.len() as f64)
    }

    pub fn train(&mut self,
             training_data: Vec<(Array2<f64>, Array2<f64>)>,
             epochs: usize,
             batches_per_epoch: usize,
             batch_size: usize,
             learning_rate: f64,
             lambda: f64
    ) {
        let weight_size: Vec<Array2<f64>> = self.layers.iter().map(|w|Array2::zeros(w.weights.dim())).collect();
        let bias_size: Vec<Array2<f64>> = self.layers.iter().map(|w|Array2::zeros(w.biases.dim())).collect();
        let mut activations_shape: Vec<Array2<f64>> = weight_size.iter().map(|w| Array2::zeros((w.dim().1, 1))).collect();

        activations_shape.push(Array2::zeros((self.layers.last().unwrap().biases.dim().0, 1)));

        let inputs = training_data.iter().map(|(i, _)| i.clone()).collect::<Vec<_>>();
        let truths = training_data.iter().map(|(_, t)| t.clone()).collect::<Vec<_>>();

        let mut training_data = TrainingData {
            nabla_w: weight_size.clone(),
            nabla_b: bias_size.clone(),
            delta_nabla_w: weight_size,
            delta_nabla_b: bias_size,
            activations: activations_shape.clone(),
            post_biases: activations_shape,
        };
        
        let mut batches: Vec<(Array2<f64>, Array2<f64>)> = inputs.clone().into_iter().zip(truths.clone()).collect();
        let validation_inputs: Vec<Array2<f64>> = inputs.into_iter().take(10).collect();
        let validation_truths: Vec<Array2<f64>> = truths.into_iter().take(10).collect();
        for epoch in 0..epochs {
            batches.shuffle(&mut rand::thread_rng());

            let start = std::time::Instant::now();
            
            for batch in batches.chunks(batch_size).cycle().take(batches_per_epoch) {
                self.update_weights_biases(batch, learning_rate, lambda, &mut training_data);
            }

            #[cfg(feature = "logging")]
            {
                let elapsed = start.elapsed();
                let loss = self.validate(&validation_inputs, &validation_truths);
                println!("Epoch {epoch} - Loss: {loss} - Time: {elapsed:?}");
                if loss.is_nan() {
                    panic!("Loss is NaN. Training aborted.");
                }
            }
        }
    }

    fn update_weights_biases(&mut self,
                             batch: &[(Array2<f64>, Array2<f64>)],
                             learning_rate: f64,
                             lambda: f64,
                             training_data: &mut TrainingData
    ) {
        let nabla_w = &mut training_data.nabla_w;
        let nabla_b = &mut training_data.nabla_b;
        let delta_nabla_w = &mut training_data.delta_nabla_w;
        let delta_nabla_b = &mut training_data.delta_nabla_b;
        let activations = &mut training_data.activations;
        let post_biases = &mut training_data.post_biases;
        reset_matrix(nabla_w);
        reset_matrix(nabla_b);
        reset_matrix(delta_nabla_w);
        reset_matrix(delta_nabla_b);
        for (o, t) in batch {
            self.backpropagation(o, t, delta_nabla_w, delta_nabla_b, activations, post_biases);
            for (nw, dnw) in nabla_w.iter_mut().zip(delta_nabla_w.iter()) {
                *nw += dnw;
            }
            for (nb, dnb) in nabla_b.iter_mut().zip(delta_nabla_b.iter()) {
                *nb += dnb;
            }
            reset_matrix(delta_nabla_w);
            reset_matrix(delta_nabla_b);
        }

        for (w, nw) in self.layers.iter_mut().zip(nabla_w.iter()) {
            w.weights.scaled_add(-learning_rate, nw);
            if lambda != 0.0 {
                w.weights.mapv_inplace(|x| x - lambda * x);
            }
        }
        for (b, nb) in self.layers.iter_mut().zip(nabla_b.iter()) {
            b.biases.scaled_add(-learning_rate, nb);
        }
    }

    fn backpropagation(&mut self,
                       input: &Array2<f64>,
                       truth: &Array2<f64>,
                       delta_nabla_w: &mut [Array2<f64>],
                       delta_nabla_b: &mut [Array2<f64>],
                       activations: &mut [Array2<f64>],
                       post_biases: &mut Vec<Array2<f64>>,
    ) {
        let mut previous_output = input.clone(); // output from the previous layer
        activations[0].assign(&previous_output);
        // feedforward
        let mut z: Array2<f64>;
        for (i, layer) in self.layers.iter().enumerate() {
            z = layer.weights.dot(&previous_output);
            z += &layer.biases;
            post_biases[i + 1].assign(&z);
            z.mapv_inplace(layer.activation_function);
            previous_output = z;
            activations[i + 1].assign(&previous_output);
        }

        // backward pass
        let mut delta: &mut Array2<f64> = &mut cost_derivative(activations.last().unwrap(), truth);
        delta.zip_mut_with(&post_biases.pop().unwrap(), |d, s| *d *= (self.layers.last().unwrap().activation_function_prime)(*s));
        let nabla_b_len = delta_nabla_b.len();
        delta_nabla_b[nabla_b_len - 1].assign(delta);
        let nabla_w_len = delta_nabla_w.len();
        delta_nabla_w[nabla_w_len - 1] = delta.dot(&activations[activations.len() - 2].t());
        
        for (l, post_bias) in post_biases.iter_mut().enumerate().skip(1).rev() {
            post_bias.mapv_inplace(self.layers[l - 1].activation_function_prime);
            // FIXME
            let dot = self.layers[l].weights.t().dot(delta);
            delta.zip_mut_with(post_bias, |d, s| *d *= s);
            delta_nabla_b[l - 1].assign(delta);
            delta_nabla_w[l - 1] = delta.dot(&activations[l - 1].t());
        }
    }
}

pub fn print_matrix<T: Display>(matrix: &ArrayView<T, Ix2>) {
    for row in matrix.rows() {
        for cell in row {
            print!("{:.2} ", cell);
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
