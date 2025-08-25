use std::path::Path;
use anyhow::Result;
use ndarray::{Array2, arr2};
use neural_net::layers::{default_linear, default_relu};
use neural_net::loss::LossFunction::BinaryCrossEntropy;
use neural_net::neural_net::{NeuralNetwork, print_matrix};
use neural_net::saving_and_loading::{load_from_file, save_to_file, Format};
use num_traits::Pow;
use rand::rng;

fn gen_heart_dataset(low: f64, high: f64, step: f64) -> Vec<(Array2<f64>, Array2<f64>)> {
    let mut dataset = Vec::new();

    let mut x = low;
    let mut y: f64;
    // x=16sin^3(x) y=13cos(x)-5cos(2x)-2cos(3x)-cos(4x)
    while x < high {
        y = low;

        let s: f64 = x.signum() * (x / 16.0).abs().pow(1.0 / 3.0);
        let A: f64 = -6.0 + 18.0 * s.powi(2) - 8.0 * s.pow(4.0);
        let B: f64 = (11.0 + 8.0 * s.pow(2.0)) * (1.0 - s.powi(2)).sqrt();
        let upper = A + B;
        let lower = A - B;
        while y < high {
            if y <= upper && y >= lower {
                dataset.push((arr2(&[[x], [y]]), arr2(&[[1_f64]])));
                //print!("X")
            } else {
                dataset.push((arr2(&[[x], [y]]), arr2(&[[0_f64]])));
                //print!("_")
            }
            y += step;
        }
        x += step;
        println!();
    }

    dataset
}

fn main() -> Result<()> {
    //let a = load_from_file::<NeuralNetwork>(Path::new("./neural_network.nn"), Format::Binary)?;
    
    let mut n = NeuralNetwork::new(vec![
        default_relu(2, 8),
        default_relu(8, 4),
        default_relu(4, 2),
        default_linear(2, 1),
    ]);

    let start = std::time::Instant::now();

    let num_epochs: usize = 2308;
    let dataset = gen_heart_dataset(-16.0, 16.0, 0.1); // datasets::gen_circle_dataset(0.0, 5.0, 0.01);
    n.train(
        dataset,
        num_epochs,
        20,
        30,
        0.01,
        0.0,
        &mut rng(),
        BinaryCrossEntropy,
    );
    
    save_to_file(Path::new("./neural_network.json"), &n, Format::Json)?;

    let elapsed = start.elapsed();
    println!("Elapsed: {elapsed:?}");
    println!(
        "{:.2?} epochs/s",
        (num_epochs as f32) / elapsed.as_secs_f32()
    );

    println!("\nWeights:");
    for (i, l) in n.layers.iter().enumerate() {
        println!("{i}");
        print_matrix(&l.weights.view());
    }

    println!("\nBiases: ");
    for (i, l) in n.layers.iter().enumerate() {
        println!("{i}");
        print_matrix(&l.biases.view());
    }
    
    Ok(())

    //let mut i: f64 = 0.0;
    //let mut input = Array2::zeros((1, 1));
    //let mut output = Array2::zeros((1,1));
    //while i <= 10.0 {
    //    input[[0, 0]] = i;
    //    output.assign(&n.feedforward(&input));
    //    println!("{:.4?} - {} -> {}", i, i.powi(2), output[[0, 0]]);
    //    i += 0.5;
    //}
}
