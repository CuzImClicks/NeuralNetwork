#[deny(clippy::nursery)]
pub mod activation;
pub mod datasets;
pub mod layers;
pub mod loss;
pub mod neural_net;

#[cfg(test)]
mod tests;
