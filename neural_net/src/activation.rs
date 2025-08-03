use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub enum Activation {
    Sigmoid,
    ReLU,
    LeakyReLU,
    Tanh,
    Linear,
}

impl Activation {
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Self::ReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.0
                }
            }
            Self::LeakyReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.01 * x
                }
            }
            Self::Tanh => x.tanh(),
            Self::Linear => x,
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Self::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            Self::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::LeakyReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.01
                }
            }
            Self::Tanh => x.tanh().mul_add(-x.tanh(), 1.0),
            Self::Linear => 1.0,
        }
    }
}
