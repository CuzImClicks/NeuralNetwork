
#[inline(always)]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
pub fn sigmoid_prime(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

pub fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

pub fn relu_prime(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

pub fn leaky_relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.01 * x }
}

pub fn leaky_relu_prime(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.01 }
}

pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

pub fn tanh_prime(x: f64) -> f64 {
    1.0 - x.tanh().powi(2)
}