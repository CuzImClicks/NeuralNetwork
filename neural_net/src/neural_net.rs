use ndarray::{Array2, ArrayBase, ArrayView, Ix2, OwnedRepr};
use rand::prelude::SliceRandom;
use rand::Rng;
use std::fmt::Display;
use crate::layers::Layer;

pub struct NeuralNetwork {
    pub layers: Vec<Layer>
}

struct TrainingData {
    nabla_w: Vec<Array2<f64>>,
    nabla_b: Vec<Array2<f64>>,
    delta_nabla_w: Vec<Array2<f64>>,
    delta_nabla_b: Vec<Array2<f64>>,
}

fn cost_derivative(output: &ArrayBase<OwnedRepr<f64>, Ix2>, truth: &Array2<f64>) -> Array2<f64> {
    output - truth
}

fn error_squared(output: &Array2<f64>, truth: &Array2<f64>) -> f64 {
    output
        .iter()
        .zip(truth.iter())
        .map(|(o, t)| (o - t).powi(2))
        .sum::<f64>() / (output.len() as f64)
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

    pub fn validate(&self, inputs: &Vec<Array2<f64>>, truths: &Vec<Array2<f64>>) -> f64 {
        inputs
            .iter()
            .zip(truths.iter())
            .map(|(i, o)| error_squared(&self.feedforward(i), &o))
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

        let mut inputs = training_data.iter().map(|(i, _)| i.clone()).collect::<Vec<_>>();
        let mut truths = training_data.iter().map(|(_, t)| t.clone()).collect::<Vec<_>>();
        
        let mut training_data = TrainingData {
            nabla_w: weight_size.clone(),
            nabla_b: bias_size.clone(),
            delta_nabla_w: weight_size,
            delta_nabla_b: bias_size,
        };
        
        let mut batches: Vec<(Array2<f64>, Array2<f64>)> = inputs.clone().into_iter().zip(truths.clone().into_iter()).collect();
        for epoch in 0..epochs {
            inputs.shuffle(&mut rand::thread_rng());

            //let start = std::time::Instant::now();
            
            for batch in batches.chunks(batch_size).cycle().take(batches_per_epoch) {
                self.update_weights_biases(batch, learning_rate, lambda, &mut training_data);
            }
            //let elapsed = start.elapsed();

            //let loss = self.validate(&inputs, &truths);
            //println!("Epoch {epoch} - {loss} - {elapsed:?}");
            //if loss.is_nan() {
            //    panic!("Loss is NaN. Training aborted.");
            //}
        }
    }

    fn update_weights_biases(&mut self,
                             batch: &[(Array2<f64>, Array2<f64>)],
                             learning_rate: f64,
                             lambda: f64,
                             training_data: &mut TrainingData
    ) {
        let mut nabla_w = &mut training_data.nabla_w;
        let mut nabla_b = &mut training_data.nabla_b;
        let mut delta_nabla_w = &mut training_data.delta_nabla_w;
        let mut delta_nabla_b = &mut training_data.delta_nabla_b;
        reset_matrix(&mut nabla_w);
        reset_matrix(&mut nabla_b);
        reset_matrix(&mut delta_nabla_w);
        reset_matrix(&mut delta_nabla_b);
        for (o, t) in batch {
            self.backpropagation(o, t, &mut delta_nabla_w, &mut delta_nabla_b);
            for (nw, dnw) in nabla_w.iter_mut().zip(delta_nabla_w.iter()) {
                *nw += dnw;
            }
            for (nb, dnb) in nabla_b.iter_mut().zip(delta_nabla_b.iter()) {
                *nb += dnb;
            }
            reset_matrix(&mut delta_nabla_w);
            reset_matrix(&mut delta_nabla_b);
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

    fn backpropagation(&mut self, input: &Array2<f64>, truth: &Array2<f64>, nabla_w: &mut [Array2<f64>], nabla_b: &mut [Array2<f64>]) {
        let num_layers = self.layers.len();
        let mut activation = input.clone();
        let mut activations: Vec<Array2<f64>> = Vec::with_capacity(num_layers + 1);
        activations.push(activation.clone());
        let mut zs = Vec::with_capacity(num_layers);

        // feedforward
        for layer in self.layers.iter() {
            let mut z = layer.weights.dot(&activation);
            z += &layer.biases;
            zs.push(z.clone());
            z.mapv_inplace(layer.activation_function);
            activation = z;
            activations.push(activation.clone());
        }

        // backward pass
        let mut delta = cost_derivative(activations.last().unwrap(), truth);
        delta.zip_mut_with(&zs.pop().unwrap(), |d, s| *d *= (self.layers.last().unwrap().activation_function_prime)(*s));
        let nabla_b_len = nabla_b.len();
        nabla_b[nabla_b_len - 1].assign(&delta);
        let nabla_w_len = nabla_w.len();
        nabla_w[nabla_w_len - 1] = delta.dot(&activations[activations.len() - 2].t());

        let mut z: Array2<f64> = Array2::zeros(zs.first().unwrap().dim());
        for l in (1..self.layers.len()).rev() {
            z = zs.pop().unwrap();
            z.mapv_inplace(self.layers[l - 1].activation_function_prime);
            delta = self.layers[l].weights.t().dot(&delta) * &z;
            nabla_b[l - 1].assign(&delta);
            nabla_w[l - 1] = delta.dot(&activations[l - 1].t());
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

fn mirror_zeros<O: num_traits::Zero + Clone>(i: &[Array2<O>]) -> Vec<Array2<O>> {
    i.iter().map(|x| Array2::zeros(x.dim())).collect()
}

#[inline(always)]
pub fn reset_matrix<O: num_traits::Zero + Copy>(i: &mut [Array2<O>]) {
    for x in i.iter_mut() {
        x.fill(O::zero());
    }
}
