use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub enum LossFunction {
    MeanErrorSquared,
    BinaryCrossEntropy,
}

impl LossFunction {
    pub fn apply(&self, output: ArrayView2<f64>, truth: ArrayView2<f64>) -> f64 {
        match self {
            LossFunction::MeanErrorSquared => {
                let diff = &output - &truth;
                let m = output.len() as f64;
                diff.mapv(|d| d.powi(2)).sum() / (2.0 * m)
            }
            LossFunction::BinaryCrossEntropy => {
                let eps = 1e-12;
                let o = output.mapv(|v| v.clamp(eps, 1.0 - eps));
                let y = truth;
                let m = o.len() as f64;
                -((&y * o.mapv(|v| v.ln()) + &(1.0 - &y) * (1.0 - &o).mapv(|v| v.ln())).sum()) / m
            }
        }
    }

    pub fn derivative(&self, output: ArrayView2<f64>, truth: ArrayView2<f64>) -> Array2<f64> {
        match self {
            LossFunction::MeanErrorSquared => (&output - &truth) / (output.len() as f64),
            LossFunction::BinaryCrossEntropy => (&output - &truth) / (output.len() as f64),
        }
    }
}
