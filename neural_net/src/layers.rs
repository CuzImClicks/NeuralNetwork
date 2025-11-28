use cubecl::server::Allocation;
use ndarray::Array2;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::activation::Activation;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Layer {
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
    pub activation: Activation,
}

#[derive(Debug)]
pub struct GpuLayer {
    pub weights: Allocation,
    pub biases: Allocation,
    pub activation: Activation,
}

impl Layer {
    pub fn new(weights: Array2<f64>, biases: Array2<f64>, activation: Activation) -> Self {
        Layer {
            weights,
            biases,
            activation,
        }
    }

    pub fn initialise_with(
        layer_sizes: (usize, usize),
        weight_fn: fn(usize, usize) -> Array2<f64>,
        bias_fn: fn(usize, usize) -> Array2<f64>,
        activation: Activation,
    ) -> Self {
        Layer {
            weights: weight_fn(layer_sizes.0, layer_sizes.1),
            biases: bias_fn(1, layer_sizes.1),
            activation,
        }
    }
}

pub fn default_sigmoid(input_size: usize, output_size: usize) -> Layer {
    Layer::initialise_with(
        (input_size, output_size),
        gen_xavier_matrix,
        gen_zero_matrix,
        Activation::Sigmoid,
    )
}

pub fn default_relu(input_size: usize, output_size: usize) -> Layer {
    Layer::initialise_with(
        (input_size, output_size),
        gen_he_matrix,
        gen_zero_matrix,
        Activation::ReLU,
    )
}

pub fn default_linear(input_size: usize, output_size: usize) -> Layer {
    Layer::initialise_with(
        (input_size, output_size),
        gen_random_matrix,
        gen_zero_matrix,
        Activation::Linear,
    )
}

pub fn default_tanh(input_size: usize, output_size: usize) -> Layer {
    Layer::initialise_with(
        (input_size, output_size),
        gen_xavier_matrix,
        gen_zero_matrix,
        Activation::Tanh,
    )
}

pub fn default_leaky_relu(input_size: usize, output_size: usize) -> Layer {
    Layer::initialise_with(
        (input_size, output_size),
        gen_he_matrix,
        gen_zero_matrix,
        Activation::LeakyReLU,
    )
}

pub fn gen_random_matrix(input_size: usize, output_size: usize) -> Array2<f64> {
    let mut rng = rand::rng();
    Array2::from_shape_fn((output_size, input_size), |_| rng.random_range(-1.0..1.0))
}

pub fn gen_he_matrix(input_size: usize, output_size: usize) -> Array2<f64> {
    let mut rng = rand::rng();
    let variance = (2.0 / (input_size as f64)).sqrt();
    Array2::from_shape_fn((output_size, input_size), |_| {
        rng.random_range(-variance..variance)
    })
}

pub fn gen_xavier_matrix(input_size: usize, output_size: usize) -> Array2<f64> {
    let mut rng = rand::rng();
    let variance = (2.0 / (input_size + output_size) as f64).sqrt();
    Array2::from_shape_fn((output_size, input_size), |_| {
        rng.random_range(-variance..variance)
    })
}

pub fn gen_zero_matrix(input_size: usize, output_size: usize) -> Array2<f64> {
    Array2::zeros((output_size, input_size))
}
