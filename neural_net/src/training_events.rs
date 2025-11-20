use crate::neural_net::NeuralNetwork;
#[cfg(feature = "loss")]
use std::time::Duration;
use std::time::Instant;

#[derive(Debug, Copy, Clone)]
pub struct EpochStats {
    pub epoch: usize,
    #[cfg(feature = "loss")]
    pub loss: f64,
    #[cfg(feature = "loss")]
    pub loss_elapsed: Duration,
    pub backpropagation_elapsed: Duration,
}

#[derive(Debug, Copy, Clone)]
pub enum TrainingEvent {
    TrainingBegin {
        start_time: Instant,
        total_epochs: usize,
    },
    EpochBegin {
        epoch: usize,
    },
    EpochEnd {
        stats: EpochStats,
    },
    TrainingEnd {
        end_time: Duration,
        total_epochs: usize,
    },
}

pub trait TrainingCallback {
    fn on_event(&mut self, _nn: &NeuralNetwork, _event: TrainingEvent);
}

pub struct Logger;

impl TrainingCallback for Logger {
    fn on_event(&mut self, _nn: &NeuralNetwork, event: TrainingEvent) {
        match event {
            TrainingEvent::TrainingBegin {
                start_time: _start_time,
                total_epochs,
            } => {
                log::info!("Started training for `{}` epochs", total_epochs);
            }
            TrainingEvent::EpochEnd { stats } => {
                #[cfg(not(feature = "loss"))]
                {
                    let epoch = stats.epoch + 1;
                    let elapsed = stats.backpropagation_elapsed;

                    log::info!("Epoch {epoch} - Time: {elapsed:?}");
                }

                #[cfg(feature = "loss")]
                {
                    log::info!(
                        "Epoch {} - Loss: {} - Time: {:?} | {:?}",
                        stats.epoch + 1,
                        stats.loss,
                        stats.backpropagation_elapsed,
                        stats.loss_elapsed
                    );
                    if stats.loss.is_nan() {
                        panic!("Loss is NaN. Training aborted.");
                    }
                }
            }
            TrainingEvent::TrainingEnd {
                end_time,
                total_epochs,
            } => {
                log::info!("Finished training in {:?}s", end_time.as_secs());
                log::info!("{}/s", total_epochs / end_time.as_secs() as usize);
            }
            _ => {}
        }
    }
}

pub struct LossCollector {
    pub data: Vec<(f64, f64)>,
}

impl LossCollector {
    pub fn new(epochs: usize) -> Self {
        Self {
            data: Vec::with_capacity(epochs),
        }
    }
}

impl TrainingCallback for LossCollector {
    fn on_event(&mut self, _nn: &NeuralNetwork, event: TrainingEvent) {
        if let TrainingEvent::EpochEnd { stats } = event {
            self.data.push((stats.epoch as f64, stats.loss));
        }
    }
}

#[derive(Default)]
pub struct Callbacks<'a>(Vec<&'a mut dyn TrainingCallback>);

impl<'a> Callbacks<'a> {
    pub fn new(callbacks: Vec<&'a mut dyn TrainingCallback>) -> Self {
        Callbacks(callbacks)
    }

    pub fn on_event(&mut self, nn: &NeuralNetwork, event: TrainingEvent) {
        if !self.0.is_empty() {
            self.0.iter_mut().for_each(|it| it.on_event(nn, event));
        }
    }
}
