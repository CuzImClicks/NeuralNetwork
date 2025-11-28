use std::{
    fs,
    path::Path,
    time::{Duration, Instant},
};

use log::warn;

use crate::{
    datasets::Float, neural_net::NeuralNetwork, saving_and_loading::Format, training_events::{TrainingCallback, TrainingEvent}
};

#[derive(Debug)]
pub enum CheckpointStrategy<'a, T: AsRef<Path>> {
    Percentage {
        percentage: Float,
        total_epochs: usize,
        folder: T,
    },
    TimePassed {
        wait_time: Duration,
        start_time: Instant,
        last_save: Instant,
        folder: T,
    },
    LowestLoss {
        folder: T,
        save_on_training_end: bool,
        neural_network: Option<NeuralNetwork>,
        lowest_loss: &'a mut Float,
        epoch: &'a mut usize,
    },
}

impl<'a, T: AsRef<Path>> TrainingCallback for CheckpointStrategy<'a, T> {
    fn on_event(&mut self, nn: &NeuralNetwork, event: &TrainingEvent) {
        match self {
            CheckpointStrategy::Percentage {
                percentage,
                total_epochs,
                folder,
            } => match event {
                TrainingEvent::EpochEnd { stats } => {
                    let a = ((*total_epochs as Float) * *percentage).round();
                    let completion_percentage =
                        ((stats.epoch as Float / *total_epochs as Float) * 100.0).round();
                    if stats.epoch as Float % a == 0.0 {
                        nn.save_checkpoint(
                            folder
                                .as_ref()
                                .join(format!("checkpoint_{}.json", completion_percentage)),
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
                                .join(format!("checkpoint_{}.json", stats.epoch)),
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
            CheckpointStrategy::LowestLoss {
                lowest_loss,
                epoch,
                folder,
                neural_network,
                save_on_training_end,
            } => match event {
                TrainingEvent::EpochEnd { stats } => {
                    if stats.loss <= **lowest_loss {
                        *neural_network = Some(nn.clone());
                        **lowest_loss = stats.loss;
                        **epoch = stats.epoch;
                        if *save_on_training_end {
                            nn.save_checkpoint(
                                folder
                                    .as_ref()
                                    .join(format!("checkpoint_{}.json", stats.epoch)),
                                Format::Json,
                            )
                            .unwrap();
                            warn!("Saved checkpoint due to lowest loss `{:?}`", lowest_loss)
                        }
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
                TrainingEvent::TrainingEnd {
                    end_time: _end_time,
                    total_epochs: _total_epochs,
                    training_dataset: _,
                    validation_dataset: _,
                } => {
                    if !*save_on_training_end {
                        nn.save_checkpoint(
                            folder.as_ref().join(format!("checkpoint_{}.json", epoch)),
                            Format::Json,
                        )
                        .unwrap();
                        warn!(
                            "Saved checkpoint at the end of training with lowest loss `{:?}`",
                            lowest_loss
                        )
                    }
                }
                _ => {}
            },
        }
    }
}
