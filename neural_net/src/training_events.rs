#[cfg(feature = "loss")]
use std::time::Duration;
use std::{fs, path::Path, time::Instant};

use log::warn;

use crate::{neural_net::NeuralNetwork, saving_and_loading::Format};

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
        match event {
            TrainingEvent::EpochEnd { stats } => {
                self.data.push((stats.epoch as f64, stats.loss));
            }
            _ => {}
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CheckpointStrategy<T: AsRef<Path>> {
    Percentage {
        percentage: f64,
        total_epochs: usize,
        folder: T,
    },
    TimePassed {
        wait_time: Duration,
        start_time: Instant,
        last_save: Instant,
        folder: T,
    },
}

impl<T: AsRef<Path>> TrainingCallback for CheckpointStrategy<T> {
    fn on_event(&mut self, nn: &NeuralNetwork, event: TrainingEvent) {
        match self {
            CheckpointStrategy::Percentage {
                percentage,
                total_epochs,
                folder,
            } => match event {
                TrainingEvent::EpochEnd { stats } => {
                    let a = ((*total_epochs as f64) * *percentage).round();
                    let completion_percentage =
                        ((stats.epoch as f64 / *total_epochs as f64) * 100.0).round();
                    if stats.epoch as f64 % a == 0.0 {
                        nn.save_checkpoint(
                            folder
                                .as_ref()
                                .join(&format!("checkpoint_{}.json", completion_percentage)),
                            Format::Json,
                        )
                        .unwrap();
                        warn!(
                            "Saved percentage based checkpoint at {}% of finished training.",
                            completion_percentage
                        )
                    }
                }
                TrainingEvent::TrainingBegin {
                    start_time: _start_time,
                    total_epochs: _total_epochs,
                } => {
                    if !folder.as_ref().exists() {
                        fs::create_dir(folder).unwrap()
                    }
                }
                _ => {}
            },
            CheckpointStrategy::TimePassed {
                wait_time,
                start_time,
                last_save,
                folder,
            } => match event {
                TrainingEvent::EpochEnd { stats } => {
                    let duration_since = last_save.duration_since(*start_time);
                    if duration_since >= *wait_time {
                        *last_save = Instant::now();

                        nn.save_checkpoint(
                            folder
                                .as_ref()
                                .join(&format!("checkpoint_{}.json", stats.epoch)),
                            Format::Json,
                        )
                        .unwrap();
                        warn!("Saved time based checkpoint after {:?}", duration_since)
                    }
                }
                TrainingEvent::TrainingBegin {
                    start_time: _start_time,
                    total_epochs: _total_epochs,
                } => {
                    if !folder.as_ref().exists() {
                        fs::create_dir(folder).unwrap()
                    }
                }
                _ => {}
            },
        }
    }
}

pub struct Callbacks<'a>(Vec<&'a mut dyn TrainingCallback>);

impl<'a> Callbacks<'a> {
    pub fn new(callbacks: Vec<&'a mut dyn TrainingCallback>) -> Self {
        Callbacks(callbacks)
    }

    pub fn on_event(&mut self, nn: &NeuralNetwork, event: TrainingEvent) {
        if !self.0.is_empty() {
            self.0.iter_mut().for_each(|it| it.on_event(&nn, event));
        }
    }
}

impl<'a> Default for Callbacks<'a> {
    fn default() -> Self {
        Self(vec![])
    }
}
