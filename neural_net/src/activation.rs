
#[inline(always)]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
pub fn sigmoid_prime(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

#[inline(always)]
pub fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

#[inline(always)]
pub fn relu_prime(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

#[inline(always)]
pub fn leaky_relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.01 * x }
}

#[inline(always)]
pub fn leaky_relu_prime(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.01 }
}

#[inline(always)]
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

#[inline(always)]
pub fn tanh_prime(x: f64) -> f64 {
    1.0 - x.tanh().powi(2)
}

#[inline(always)]
pub fn linear(x: f64) -> f64 {
    x
}

#[inline(always)]
pub fn linear_prime(_x: f64) -> f64 {
    1.0
}