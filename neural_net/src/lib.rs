#[deny(clippy::nursery)]
pub mod activation;
pub mod datasets;
pub mod layers;
pub mod loss;
pub mod neural_net;
pub mod saving_and_loading;

#[cfg(test)]
mod tests;
