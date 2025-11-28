use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub name: String,
    pub creation_date: NaiveDateTime,
    pub last_modified_date: NaiveDateTime,
    pub training_history: Vec<TrainingRun>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRun {
    pub epochs: usize,
    pub start_time: NaiveDateTime,
    pub stop_time: NaiveDateTime,
}
