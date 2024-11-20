#![feature(iter_next_chunk)]

mod activation;

use std::fmt::Display;
use ndarray::{arr2, Array2, ArrayBase, Ix2, OwnedRepr};
use rand::prelude::SliceRandom;
use rand::Rng;
use crate::activation::{leaky_relu, leaky_relu_prime};

struct NeuralNetwork {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array2<f64>>,
}

fn gen_random_matrix(rows: usize, cols: usize) -> Array2<f64>
{
    let mut rng = rand::thread_rng();
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-1.0..1.0))
}

fn cost_derivative(output: &ArrayBase<OwnedRepr<f64>, Ix2>, truth: &Array2<f64>) -> Array2<f64> {
    output - truth
}

fn mean_squared_error(output: &[f64], truth: &[f64]) -> f64 {
    output
        .iter()
        .zip(truth.iter())
        .map(|(o, t)| (o - t).powi(2))
        .sum::<f64>() / (output.len() as f64)
}

impl NeuralNetwork {
    fn new(weights: Vec<Array2<f64>>, biases: Vec<Array2<f64>>) -> Self {
        NeuralNetwork { weights, biases }
    }

    fn initialise_random(layer_sizes: Vec<usize>) -> Self {
        let weights  = (1..layer_sizes.len()).map(|i| gen_random_matrix(layer_sizes[i], layer_sizes[i - 1])).collect();
        let biases = (1..layer_sizes.len()).map(|i| gen_random_matrix(layer_sizes[i], 1)).collect();
        NeuralNetwork::new(weights, biases)
    }

    fn feedforward(&self, inputs: &Array2<f64>) -> Array2<f64> {
        let mut result = inputs.to_owned();
        for (layer, bias) in self.weights.iter().zip(self.biases.iter()) {
            result = layer.dot(&result);
            result += bias;
            result.mapv_inplace(leaky_relu);
        }
        result
    }

    fn validate(&self, validation_data: &[(Array2<f64>, f64)]) -> f64 {
        let truth: Vec<f64> = validation_data.iter().map(|(_, t)| *t).collect();
        let output: Vec<f64> = validation_data
            .iter()
            .map(|(i, _)| self.feedforward(i)[[0, 0]])
            .collect();

        mean_squared_error(&output, &truth)
    }

    fn train(&mut self,
             training_data: Vec<(Array2<f64>, Array2<f64>)>,
             epochs: usize,
             batch_size: usize,
             learning_rate: f64
    ) {
        let mut training_data = training_data;
        let mut nabla_w: Vec<Array2<f64>> = self.weights.iter().map(|w|Array2::zeros(w.dim())).collect();
        let mut nabla_b: Vec<Array2<f64>> = self.biases.iter().map(|w|Array2::zeros(w.dim())).collect();
        let mut delta_nabla_w: Vec<Array2<f64>> = self.weights.iter().map(|w|Array2::zeros(w.dim())).collect();
        let mut delta_nabla_b: Vec<Array2<f64>> = self.biases.iter().map(|w|Array2::zeros(w.dim())).collect();

        let validation: Vec<(ArrayBase<OwnedRepr<f64>, Ix2>, f64)> = training_data.iter().map(|x| (x.0.clone(), x.1[(0,0)].clone())).collect();
        for epoch in 0..epochs {
            training_data.shuffle(&mut rand::thread_rng());
            let loss = self.validate(&validation);
            println!("Epoch {epoch} - {loss}");
            if loss.is_nan() {
                panic!("loss is nan");
            }
            for batch in training_data.chunks(batch_size) {
                self.update_weights_biases(batch, learning_rate, &mut nabla_w, &mut nabla_b, &mut delta_nabla_w, &mut delta_nabla_b);
            }
        }
    }

    fn update_weights_biases(&mut self,
                             batch: &[(Array2<f64>, Array2<f64>)],
                             learning_rate: f64,
                             nabla_w: &mut [Array2<f64>],
                             nabla_b: &mut [Array2<f64>],
                             delta_nabla_w: &mut [Array2<f64>],
                             delta_nabla_b: &mut [Array2<f64>],
    ) {
        reset_matrix(nabla_w);
        reset_matrix(nabla_b);
        reset_matrix(delta_nabla_w);
        reset_matrix(delta_nabla_b);
        for (o, t) in batch {
            self.backpropagation(o, t, delta_nabla_w, delta_nabla_b);
            for (nw, dnw) in nabla_w.iter_mut().zip(delta_nabla_w.iter()) {
                *nw += dnw;
            }
            for (nb, dnb) in nabla_b.iter_mut().zip(delta_nabla_b.iter()) {
                *nb += dnb;
            }
            reset_matrix(delta_nabla_w);
            reset_matrix(delta_nabla_b);
        }

        let batch_size = batch.len() as f64;
        let l_rate = learning_rate / batch_size;
        for (w, nw) in self.weights.iter_mut().zip(nabla_w.iter()) {
            w.scaled_add(-l_rate, nw);
        }
        for (b, nb) in self.biases.iter_mut().zip(nabla_b.iter()) {
            b.scaled_add(-l_rate, nb);
        }
    }

    fn backpropagation(&mut self, input: &Array2<f64>, truth: &Array2<f64>, nabla_w: &mut [Array2<f64>], nabla_b: &mut [Array2<f64>]) {
        let num_layers = self.weights.len();
        let mut activation = input.clone();
        let mut activations: Vec<Array2<f64>> = Vec::with_capacity(num_layers + 1);
        activations.push(activation.clone());
        let mut zs = Vec::with_capacity(num_layers);

        // feedforward
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            let mut z = w.dot(&activation);
            z += b;
            zs.push(z.clone());
            z.mapv_inplace(leaky_relu);
            activation = z;
            activations.push(activation.clone());
        }

        // backward pass
        let mut delta = cost_derivative(activations.last().unwrap(), truth);
        delta.zip_mut_with(&zs.pop().unwrap(), |d, s| *d *= leaky_relu_prime(*s));
        let nabla_b_len = nabla_b.len();
        nabla_b[nabla_b_len - 1].assign(&delta);
        let nabla_w_len = nabla_w.len();
        nabla_w[nabla_w_len - 1] = delta.dot(&activations[activations.len() - 2].t());

        for l in (1..self.weights.len()).rev() {
            let mut z = zs.pop().unwrap();
            z.mapv_inplace(leaky_relu_prime);
            delta = self.weights[l].t().dot(&delta) * z;
            nabla_b[l - 1].assign(&delta);
            nabla_w[l - 1] = delta.dot(&activations[l - 1].t());
        }
    }
}

fn print_matrix<T: Display>(matrix: &Array2<T>) {
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

fn reset_matrix<O: num_traits::Zero + Clone>(i: &mut [Array2<O>]) {
    for x in i.iter_mut() {
        x.fill(O::zero());
    }
}

fn gen_x2_dataset(low: f64, high: f64, step: f64) -> Vec<(Array2<f64>, Array2<f64>)> {

    let mut v: Vec<(Array2<f64>, Array2<f64>)> = Vec::with_capacity((high / step) as usize);
    let mut i = low;
    while i <= high {
        v.push((arr2(&[[i]]), arr2(&[[i.powi(2)]])));
        i += step;
    }
    v
}

fn gen_xor_dataset() -> Vec<(Array2<f64>, Array2<f64>)> {
    vec![
        (arr2(&[[0.0], [0.0]]), arr2(&[[0.0]])),
        (arr2(&[[0.0], [1.0]]), arr2(&[[1.0]])),
        (arr2(&[[1.0], [0.0]]), arr2(&[[1.0]])),
        (arr2(&[[1.0], [1.0]]), arr2(&[[0.0]])),
    ]
}

fn main() {
    let mut n = NeuralNetwork::initialise_random(vec![1, 4, 4, 4, 1]);

    let start = std::time::Instant::now();

    let num_epochs: usize = 2000;
    let dataset = gen_x2_dataset(0.0, 10.0, 0.1);
    n.train(dataset.clone(), num_epochs, 5, 0.01);

    let elapsed = start.elapsed();
    println!("Elapsed: {:?}", elapsed);
    println!("{:.2?} epochs/s", (num_epochs as f32) / elapsed.as_secs_f32());

    println!("\nWeights:");
    for (i, w) in n.weights.iter().enumerate() {
        println!("{}", i);
        print_matrix(w);
    }

    println!("\nBiases: ");
    for (i, b) in n.biases.iter().enumerate() {
        println!("{}", i);
        print_matrix(b);
    }

    println!("\nResults:");

    for (_, (input, output)) in dataset.iter().enumerate() {
        let result = n.feedforward(&input);
        print!("Input: ");
        print_matrix(input);
        print!("Expected: ");
        print_matrix(output);
        print!("Got: ");
        print_matrix(&result);
        println!();
    }

    let mut i = 0.0;
    while i < 10.0 {
        let result = n.feedforward(&arr2(&[[i]]));
        println!("{} -> {}", i, result[[0, 0]]);
        i += 0.1;
    }
}
