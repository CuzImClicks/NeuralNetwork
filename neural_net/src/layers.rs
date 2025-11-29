use cubecl::{Runtime, server::Allocation};
use ndarray::Array2;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{activation::Activation, datasets::Float, gpu::gpu_tensor::GpuTensor};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Layer {
    pub weights: Array2<Float>,
    pub biases: Array2<Float>,
    pub activation: Activation,
}

#[cfg(feature = "gpu")]
#[derive(Debug)]
pub struct GpuLayer<R: Runtime> {
    pub weights: GpuTensor<R, Float>,
    pub biases: GpuTensor<R, Float>,
    pub activation: Activation,
}

impl Layer {
    pub fn new(weights: Array2<Float>, biases: Array2<Float>, activation: Activation) -> Self {
        Layer {
            weights,
            biases,
            activation,
        }
    }

    pub fn initialise_with(
        layer_sizes: (usize, usize),
        weight_fn: fn(usize, usize) -> Array2<Float>,
        bias_fn: fn(usize, usize) -> Array2<Float>,
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

pub fn gen_random_matrix(input_size: usize, output_size: usize) -> Array2<Float> {
    let mut rng = rand::rng();
    Array2::from_shape_fn((output_size, input_size), |_| rng.random_range(-1.0..1.0))
}

pub fn gen_he_matrix(input_size: usize, output_size: usize) -> Array2<Float> {
    let mut rng = rand::rng();
    let variance = (2.0 / (input_size as Float)).sqrt();
    Array2::from_shape_fn((output_size, input_size), |_| {
        rng.random_range(-variance..variance)
    })
}

pub fn gen_xavier_matrix(input_size: usize, output_size: usize) -> Array2<Float> {
    let mut rng = rand::rng();
    let variance = (2.0 / (input_size + output_size) as Float).sqrt();
    Array2::from_shape_fn((output_size, input_size), |_| {
        rng.random_range(-variance..variance)
    })
}

pub fn gen_zero_matrix(input_size: usize, output_size: usize) -> Array2<Float> {
    Array2::zeros((output_size, input_size))
}
