#[deny(clippy::nursery)]
pub mod activation;
pub mod checkpoints;
pub mod datasets;
pub mod layers;
pub mod loss;
pub mod metadata;
pub mod neural_net;
pub mod saving_and_loading;
pub mod training_events;
pub mod training_data;
pub mod gpu;

const LINE_SIZE: u8 = 1;

#[cfg(test)]
mod tests;
