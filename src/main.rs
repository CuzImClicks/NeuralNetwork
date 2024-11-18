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
        let mut result = inputs.to_owned();
        for (layer, bias) in self.weights.iter().zip(self.biases.iter()) {
            result = layer.dot(&result);
            result += bias;
            result.mapv_inplace(sigmoid);
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
        let mut nabla_w: Vec<Array2<f64>> = self.weights.iter().map(|w|Array2::zeros(w.dim())).collect();
        let mut nabla_b: Vec<Array2<f64>> = self.biases.iter().map(|w|Array2::zeros(w.dim())).collect();
        let mut delta_nabla_w: Vec<Array2<f64>> = self.weights.iter().map(|w|Array2::zeros(w.dim())).collect();
        let mut delta_nabla_b: Vec<Array2<f64>> = self.biases.iter().map(|w|Array2::zeros(w.dim())).collect();
        for _ in 0..epochs {
            training_data.shuffle(&mut rand::thread_rng());
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
            z.mapv_inplace(sigmoid);
            activation = z;
            activations.push(activation.clone());
        }

        // backward pass
        let mut delta = cost_derivative(activations.last().unwrap().clone(), truth);
        delta.zip_mut_with(&zs.last().unwrap(), |d, s| *d *= sigmoid_prime(*s));
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
    let mut n = NeuralNetwork::new(
        vec![
            arr2(&[[1.34, -2.04], [-5.95, 5.93], [5.30, -5.39]]),
            arr2(&[[1.17, 8.76, 8.29]]),
        ],
        vec![
            arr2(&[[-1.01], [-3.34], [-3.00]]),
            arr2(&[[-4.52]]),
        ]
    ); // initialise_random(vec![2, 3, 1]);

    let start = std::time::Instant::now();

    n.stochastic_gradient_descent(gen_xor_dataset(), 100000, 5, 0.1);

    let elapsed = start.elapsed();
    println!("Elapsed: {:?}", elapsed);
    println!("{:.2?} epochs/s", 100000f32 / elapsed.as_secs_f32());

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
