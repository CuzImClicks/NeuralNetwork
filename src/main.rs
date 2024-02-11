#![feature(iter_next_chunk)]

use std::fmt::Display;
use ndarray::{arr2, Array, Array2, ArrayBase, Ix2, OwnedRepr};
use rand::prelude::SliceRandom;
use rand::Rng;

struct NeuralNetwork {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array2<f64>>,
}

// np.zeros => ndarray::Array2::zeros()

fn gen_random_matrix(rows: usize, cols: usize) -> Array2<f64>
{
    let mut rng = rand::thread_rng();
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(0.0..1.0))
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_vec(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(sigmoid)
}

fn sigmoid_prime(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
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
            result = sigmoid_vec(&(&layer.dot(&result) + bias));
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
            for batch in training_data.windows(batch_size) {
                dbg!(&batch);
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

    fn backpropagation(&mut self, output: &Array2<f64>, truth: &Array2<f64>) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut nabla_w: Vec<Array2<f64>> = mirror_zeros(&self.weights);
        let mut nabla_b: Vec<Array2<f64>> = mirror_zeros(&self.biases);
        let mut activation: Array2<f64> = output.clone();
        let mut activations: Vec<Array2<f64>> = vec![activation.clone()];
        let mut zs = vec![];

        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            let z = w.dot(&activation) + b;
            zs.push(z.clone());
            activation = z.mapv(sigmoid);
            activations.push(activation.clone());
        }

        let delta = cost_derivative(activations.last().unwrap().clone(), truth) * zs.last().unwrap().mapv(sigmoid_prime);
        let nabla_b_len = nabla_b.len();
        nabla_b[nabla_b_len - 1usize] = delta.clone();

        let nabla_w_len = nabla_w.len();
        nabla_w[nabla_w_len - 1usize] = delta.dot(&activations[activations.len() - 2].t());

        for l in 2..self.weights.len() {
            let z = &zs[zs.len() - l];
            let sp = z.mapv(sigmoid_prime);
            let delta = self.weights[self.weights.len() - l + 1].t().dot(&delta) * sp;
            nabla_b[nabla_b_len - l] = delta.clone();
            nabla_w[nabla_w_len - l] = delta.dot(&activations[activations.len() - l - 1].t());
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


fn main() {
    let mut n = NeuralNetwork::initialise_random(vec![2, 3, 1]);
    //let n = NeuralNetwork {
    //    weights: vec![arr2(&[[0.41, 0.5], [0.94, 0.57], [0.24, 0.5]]), arr2(&[[0.7, 0.8, 0.9]])],
    //    biases: vec![arr2(&[[0.55], [0.82], [0.16]]), arr2(&[[0.92]])]
    //};
    //let n = NeuralNetwork {
    //    weights: vec![arr2(&[[1.91, 2.82], [2.65, 4.16]]), arr2(&[[0.55, 0.82]])],
    //    biases: vec![arr2(&[[0.55], [0.82]]), arr2(&[[0.92]])]
    //};
    //let mut n = NeuralNetwork {
    //    weights: vec![arr2(&[[-6.48660725, -6.62301231],  [ 4.66472526,  4.68940699]]), arr2(&[[-9.6819727 , -10.03086451]])],
    //    biases: vec![arr2(&[[2.59613863], [-7.27095101]]), arr2(&[[4.87653359]])]
    //};
    //for (i, w) in n.weights.iter().enumerate() {
    //    println!("{}", i);
    //    print_matrix(w);
    //}
    //for (i, b) in n.biases.iter().enumerate() {
    //    println!("{}", i);
    //    print_matrix(b);
    //}
    //print_matrix(&n.feedforward(arr2(&[[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]])));

    n.stochastic_gradient_descent(vec![(arr2(&[[1.0], [0.0]]), arr2(&[[1.0]])),
                                       (arr2(&[[0.0], [1.0]]), arr2(&[[1.0]])),
                                       (arr2(&[[1.0], [1.0]]), arr2(&[[0.0]])),
                                       (arr2(&[[0.0], [0.0]]), arr2(&[[0.0]]))], 1, 4, 0.1);

    for (i, w) in n.weights.iter().enumerate() {
        println!("{}", i);
        print_matrix(w);
    }
    for (i, b) in n.biases.iter().enumerate() {
        println!("{}", i);
        print_matrix(b);
    }

    println!("[1.0], [0.0]");
    print_matrix(&n.feedforward(arr2(&[[1.0], [0.0]])));
    println!("[0.0], [1.0]");
    print_matrix(&n.feedforward(arr2(&[[0.0], [1.0]])));
    println!("[1.0], [1.0]");
    print_matrix(&n.feedforward(arr2(&[[1.0], [1.0]])));
    println!("[0.0], [0.0]");
    print_matrix(&n.feedforward(arr2(&[[0.0], [0.0]])));

    //let a: Array2<f64> = arr2(&[[0.41, 0.50], [0.94, 0.57], [0.24, 0.50]]);
    //let b: Array2<f64> = arr2(&[[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]]);
    //let c: Array2<f64> = arr2(&[[0.24, 0.94, 0.34]]);
    //println!();
    //print_matrix(&a);
    //println!();
    //print_matrix(&b);
    //println!();
    //let d: Array2<f64> = a.dot(&b);
    //print_matrix(&d);
    //println!();
    //print_matrix(&c);
    //println!();
    //let e: Array2<f64> = c.dot(&d);
    //print_matrix(&e);
    //println!();
}
