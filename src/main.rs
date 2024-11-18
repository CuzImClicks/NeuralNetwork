#![feature(iter_next_chunk)]

mod activation;

use std::fmt::Display;
use ndarray::{arr2, Array2, ArrayBase, Ix2, OwnedRepr};
use rand::prelude::SliceRandom;
use rand::Rng;
use crate::activation::{sigmoid, sigmoid_prime};

struct NeuralNetwork {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array2<f64>>,
}

// np.zeros => ndarray::Array2::zeros()

fn gen_random_matrix(rows: usize, cols: usize) -> Array2<f64>
{
    let mut rng = rand::thread_rng();
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-1.0..1.0))
}



fn cost_derivative(output: ArrayBase<OwnedRepr<f64>, Ix2>, truth: &Array2<f64>) -> Array2<f64> {
    output - truth
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

    fn feedforward(&self, inputs: Array2<f64>) -> Array2<f64> {
        let mut result = inputs;
        for (layer, bias) in self.weights.iter().zip(self.biases.iter()) {
            result = (&layer.dot(&result) + bias).mapv(sigmoid);
        }
        result
    }

    fn stochastic_gradient_descent(&mut self,
                                   training_data: Vec<(Array2<f64>, Array2<f64>)>,
                                   epochs: usize,
                                   batch_size: usize,
                                   learning_rate: f64
    ) {
        let mut training_data = training_data;
        for _ in 0..epochs {
            training_data.shuffle(&mut rand::thread_rng());
            for batch in training_data.chunks(batch_size) {
                self.update_weights_biases(batch, learning_rate);
            }
        }
    }

    fn update_weights_biases(&mut self, batch: &[(Array2<f64>, Array2<f64>)], learning_rate: f64) {
        let mut nabla_w: Vec<Array2<f64>> = mirror_zeros(&self.weights);
        let mut nabla_b: Vec<Array2<f64>> = mirror_zeros(&self.biases);
        for (o, t) in batch {
            let (delta_nabla_w, delta_nabla_b) = self.backpropagation(o, t);
            nabla_w = nabla_w.iter().zip(delta_nabla_w).map(|(nw, dnw)| nw + dnw).collect();
            nabla_b = nabla_b.iter().zip(delta_nabla_b).map(|(nb, dnb)| nb + dnb).collect();
        }

        self.weights = self.weights.iter().zip(nabla_w.iter()).map(|(w, nw)| w - (learning_rate / batch.len() as f64) * nw).collect();
        self.biases = self.biases.iter().zip(nabla_b.iter()).map(|(b, nb)| b - (learning_rate / batch.len() as f64) * nb).collect();
    }

    fn backpropagation(&mut self, input: &Array2<f64>, truth: &Array2<f64>) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut nabla_w: Vec<Array2<f64>> = mirror_zeros(&self.weights);
        let mut nabla_b: Vec<Array2<f64>> = mirror_zeros(&self.biases);
        let mut activation = input.clone();
        let mut activations: Vec<Array2<f64>> = vec![activation.clone()];
        let mut zs = vec![];

        // feedforward
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            let z = w.dot(&activation) + b;
            zs.push(z.clone());
            activation = z.mapv(sigmoid);
            activations.push(activation.clone());
        }

        // backward pass
        let mut delta = cost_derivative(activations.last().unwrap().clone(), truth) * zs.last().unwrap().mapv(sigmoid_prime);
        let nabla_b_len = nabla_b.len();
        nabla_b[nabla_b_len - 1] = delta.clone();
        let nabla_w_len = nabla_w.len();
        nabla_w[nabla_w_len - 1] = delta.dot(&activations[activations.len() - 2].t());

        for l in (1..self.weights.len()).rev() {
            let z = &zs[l - 1];
            let sp = z.mapv(sigmoid_prime);
            delta = self.weights[l].t().dot(&delta) * sp;
            nabla_b[l - 1] = delta.clone();
            nabla_w[l - 1] = delta.dot(&activations[l - 1].t());
        }
        (nabla_w, nabla_b)
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

fn gen_x2_dataset() -> Vec<(Array2<f64>, Array2<f64>)> {
    vec![
        (arr2(&[[0.0]]), arr2(&[[0.0]])),
        (arr2(&[[0.5]]), arr2(&[[0.25]])),
        (arr2(&[[1.0]]), arr2(&[[1.0]])),
        (arr2(&[[1.5]]), arr2(&[[2.25]])),
        (arr2(&[[2.0]]), arr2(&[[4.0]])),
        (arr2(&[[2.5]]), arr2(&[[6.25]])),
        (arr2(&[[3.0]]), arr2(&[[9.0]])),
        (arr2(&[[-0.5]]), arr2(&[[0.25]])),
        (arr2(&[[-1.0]]), arr2(&[[1.0]])),
        (arr2(&[[-1.5]]), arr2(&[[2.25]])),
    ]
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
    let mut n = NeuralNetwork::initialise_random(vec![2, 3, 1]);

    let start = std::time::Instant::now();

    n.stochastic_gradient_descent(gen_xor_dataset(), 1000000, 5, 0.1);

    let elapsed = start.elapsed();
    println!("Elapsed: {:?}", elapsed);
    println!("{:.2?} epochs/s", 1000000f32 / elapsed.as_secs_f32());

    println!("\nWeights: ");
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

    for (i, (input, output)) in gen_xor_dataset().iter().enumerate() {
        let result = n.feedforward(input.clone());
        print!("Input: ");
        print_matrix(input);
        print!("Expected: ");
        print_matrix(output);
        print!("Got: ");
        print_matrix(&result);
        println!();
    }
}
