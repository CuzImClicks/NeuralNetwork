use std::path::PathBuf;

use crate::neural_net::NeuralNetwork;

pub trait Save {
    fn save(&self, path: PathBuf) -> Result<(), String>;
}

pub trait Load {
    fn load(path: PathBuf) -> Self;
}
