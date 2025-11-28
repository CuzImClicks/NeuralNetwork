use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::datasets::Float;

#[derive(Serialize, Deserialize, Debug)]
pub enum LossFunction {
    MeanErrorSquared,
    BinaryCrossEntropy,
}

impl LossFunction {
    pub fn apply(&self, output: ArrayView2<Float>, truth: ArrayView2<Float>) -> Float {
        match self {
            LossFunction::MeanErrorSquared => {
                let diff = &output - &truth;
                let m = output.len() as Float;
                diff.mapv(|d| d.powi(2)).sum() / (2.0 * m)
            }
            LossFunction::BinaryCrossEntropy => {
                let eps = 1e-12;
                let o = output.mapv(|v| v.clamp(eps, 1.0 - eps));
                let y = truth;
                let m = o.len() as Float;
                -((&y * o.mapv(|v| v.ln()) + &(1.0 - &y) * (1.0 - &o).mapv(|v| v.ln())).sum()) / m
            }
        }
    }

    pub fn derivative(&self, output: ArrayView2<Float>, truth: ArrayView2<Float>) -> Array2<Float> {
        match self {
            LossFunction::MeanErrorSquared => (&output - &truth) / (output.len() as Float),
            LossFunction::BinaryCrossEntropy => (&output - &truth) / (output.len() as Float),
        }
    }
}
