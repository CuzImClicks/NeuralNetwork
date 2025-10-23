use std::path::Path;

use crate::visualization::plot_rgb;
use anyhow::Result;
use neural_net::datasets::gen_rainbow_dataset;
use neural_net::layers::default_relu;
use neural_net::loss::LossFunction;
use neural_net::neural_net::NeuralNetwork;
use neural_net::saving_and_loading::{Format, load_from_file};
use neural_net::training_events::{Callbacks, Logger};
use rand::rng;

mod visualization;

fn main() -> Result<()> {
    env_logger::init();
    let mut n = load_from_file::<NeuralNetwork>(Path::new("./circle.bin"), Format::Binary)
        .unwrap_or_else(|_| {
            NeuralNetwork::new(vec![
                default_relu(2, 8),
                default_relu(8, 8),
                default_relu(8, 8),
                default_relu(8, 4),
                default_relu(4, 3),
            ])
        });

    let start = std::time::Instant::now();

    let num_epochs: usize = 10000;
    let dataset = gen_rainbow_dataset(-20.0, 20.0, 0.1); //gen_heart_dataset(-16.0, 16.0, 0.1); // datasets::gen_circle_dataset(0.0, 5.0, 0.01); //
    n.train(
        dataset,
        num_epochs,
        20,
        30,
        0.01,
        0.0,
        &mut rng(),
        LossFunction::BinaryCrossEntropy,
        Callbacks::new(vec![Box::new(Logger)]),
    );

    //save_to_file(Path::new("./circle.json"), &n, Format::Json)?;

    let elapsed = start.elapsed();
    println!("Elapsed: {elapsed:?}");
    println!(
        "{:.2?} epochs/s",
        (num_epochs as f32) / elapsed.as_secs_f32()
    );

    plot_rgb(
        &n,
        (-21.0, 21.0),
        (-21.0, 21.0),
        (1000, 1000),
        "heatmap.png",
    )
    .unwrap();

    Ok(())
}
