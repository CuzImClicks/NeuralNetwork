use std::fs;
use std::path::Path;

use crate::visualization::{plot_heatmap, plot_line};
use anyhow::Result;
use log::LevelFilter;
use neural_net::datasets::gen_heart_dataset;
use neural_net::layers::{
    default_relu, default_sigmoid,
};
use neural_net::loss::LossFunction;
use neural_net::neural_net::NeuralNetwork;
use neural_net::saving_and_loading::{Format, load_from_file};
use neural_net::training_events::{Callbacks, CheckpointStrategy, Logger, LossCollector};
use plotters::style::WHITE;
use plotters::style::full_palette::RED;
use rand::rng;
use std::io::Write;

mod visualization;

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(LevelFilter::Info)
        .parse_default_env()
        .format_timestamp(Some(env_logger::TimestampPrecision::Millis))
        .format(|buf, record| {
            let style = buf.default_level_style(record.level());
            let timestamp = buf.timestamp_seconds();
            writeln!(
                buf,
                "{timestamp} [{style}{}{style:#}] [{}] {}",
                record.level(),
                record.module_path().unwrap(),
                record.args(),
            )
        })
        .init();
    let mut n = load_from_file::<NeuralNetwork>(Path::new("./hsv.json"), Format::Json)
        .unwrap_or_else(|_| {
            NeuralNetwork::new(vec![
                default_relu(2, 8),
                default_relu(8, 8),
                default_relu(8, 8),
                default_relu(8, 4),
                default_sigmoid(4, 1),
            ])
        });

    let num_epochs: usize = 3000;
    let dataset = gen_heart_dataset(-16.0, 16.0, 0.1); //gen_color_dataset(-15.0, 15.0, 0.1); //gen_rainbow_dataset(-20.0, 20.0, 0.1); //gen_heart_dataset(-16.0, 16.0, 0.1); // datasets::gen_circle_dataset(0.0, 5.0, 0.01); //
    let mut loss_collector = LossCollector::new(num_epochs);
    let mut logger = Logger {};
    let mut checkpoint_strategy = CheckpointStrategy::Percentage {
        percentage: 0.1,
        total_epochs: num_epochs,
        folder: "./checkpoints/",
    };
    n.train(
        dataset,
        num_epochs,
        20,
        30,
        0.005,
        0.0,
        &mut rng(),
        LossFunction::BinaryCrossEntropy,
        Callbacks::new(vec![
            &mut logger,
            &mut loss_collector,
            &mut checkpoint_strategy,
        ]),
    );

    //save_to_file(Path::new("./hsv.json"), &n, Format::Json)?;

    //plot_rgb(
    //    &n,
    //    (-21.0, 21.0),
    //    (-21.0, 21.0),
    //    (1000, 1000),
    //    "heatmap.png",
    //)?;

    plot_heatmap(
        &n,
        (-21.0, 21.0),
        (-21.0, 21.0),
        (1000, 1000),
        "heart.png",
        WHITE,
        RED,
    )?;

    fs::read_dir("./checkpoints/")?
        .filter(|it| {
            it.as_ref()
                .is_ok_and(|v| v.path().extension().is_some_and(|ext| ext == "json"))
        })
        .for_each(|it| {
            let path = it.unwrap().path();
            let nn = load_from_file::<NeuralNetwork>(&path, Format::Json).unwrap();
            plot_heatmap(
                &nn,
                (-40.0, 40.0),
                (-40.0, 40.0),
                (1000, 1000),
                path.with_extension("png"),
                WHITE,
                RED,
            )
            .unwrap();
        });

    plot_line(
        loss_collector.data,
        "Loss",
        "loss.png",
        (1000, 1000),
        (0.0, num_epochs as f64),
        (0.0, 1.0),
    )?;
    Ok(())
}
