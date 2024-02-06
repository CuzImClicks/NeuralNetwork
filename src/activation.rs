

use std::f64::consts::E;

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

pub fn sigmoid_derivative(x: f64) -> f64 {
    // assuming the input x is already the output of the sigmoid function
    x * (1.0 - x)
}

pub fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

pub fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

