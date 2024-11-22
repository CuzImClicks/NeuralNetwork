use ndarray::Array2;
use rand::Rng;
use crate::activation::{leaky_relu, leaky_relu_prime, linear, linear_prime, relu, relu_prime, sigmoid, sigmoid_prime, tanh, tanh_prime};

#[derive(Debug)]
pub struct Layer {
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
    pub activation_function: fn(f64) -> f64,
    pub activation_function_prime: fn(f64) -> f64
}

impl Layer {
    pub fn new(weights: Array2<f64>, biases: Array2<f64>, activation_function: fn(f64) -> f64, activation_function_prime: fn(f64) -> f64) -> Self {
        Layer { weights, biases, activation_function, activation_function_prime }
    }
    
    pub fn initialise_with(
        layer_sizes: (usize, usize),
        weight_fn: fn(usize, usize) -> Array2<f64>,
        bias_fn: fn(usize, usize) -> Array2<f64>,
        activation_function: fn(f64) -> f64,
        activation_function_prime: fn(f64) -> f64
    ) -> Self {
        Layer {
            weights: weight_fn(layer_sizes.0, layer_sizes.1),
            biases: bias_fn(1, layer_sizes.1),
            activation_function,
            activation_function_prime
        }
    }
}

pub fn default_sigmoid(input_size: usize, output_size: usize) -> Layer {
    Layer::initialise_with((input_size, output_size), gen_xavier_matrix, gen_zero_matrix, sigmoid, sigmoid_prime)
}

pub fn default_relu(input_size: usize, output_size: usize) -> Layer {
    Layer::initialise_with((input_size, output_size), gen_he_matrix, gen_zero_matrix, relu, relu_prime)
}

pub fn default_linear(input_size: usize, output_size: usize) -> Layer {
    Layer::initialise_with((input_size, output_size), gen_random_matrix, gen_zero_matrix, linear, linear_prime)
}

pub fn default_tanh(input_size: usize, output_size: usize) -> Layer {
    Layer::initialise_with((input_size, output_size), gen_xavier_matrix, gen_zero_matrix, tanh, tanh_prime)
}

pub fn default_leaky_relu(input_size: usize, output_size: usize) -> Layer {
    Layer::initialise_with((input_size, output_size), gen_he_matrix, gen_zero_matrix, leaky_relu, leaky_relu_prime)
}

pub fn gen_random_matrix(input_size: usize, output_size: usize) -> Array2<f64>
{
    let mut rng = rand::thread_rng();
    Array2::from_shape_fn((output_size, input_size), |_| rng.gen_range(-1.0..1.0))
}

pub fn gen_he_matrix(input_size: usize, output_size: usize) -> Array2<f64>
{
    let mut rng = rand::thread_rng();
    let variance = (2.0 / (input_size as f64)).sqrt();
    Array2::from_shape_fn((output_size, input_size), |_| rng.gen_range(-variance..variance))
}

pub fn gen_xavier_matrix(input_size: usize, output_size: usize) -> Array2<f64>
{
    let mut rng = rand::thread_rng();
    let variance = (2.0 / (input_size + output_size) as f64).sqrt();
    Array2::from_shape_fn((output_size, input_size), |_| rng.gen_range(-variance..variance))
}

pub fn gen_zero_matrix(input_size: usize, output_size: usize) -> Array2<f64>
{
    Array2::zeros((output_size, input_size))
}


