use ndarray::{arr2, Array2};
use neural_net::layers::{default_leaky_relu, default_sigmoid};
use neural_net::neural_net::{print_matrix, NeuralNetwork};
use num_traits::Pow;
use plotters::style::RGBColor;
use rand::rng;

use crate::visualization::plot_heatmap;

mod visualization;

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
            } else {
                dataset.push((arr2(&[[x], [y]]), arr2(&[[0_f64]])));
            }
            y += step;
        }
        x += step;
    }

    dataset
}

fn inspect_predictions(nn: &NeuralNetwork, data: &[(Array2<f64>, Array2<f64>)]) {
    let mut sum = 0.0;
    let mut count = 0;
    let mut positives = 0;
    for (input, _) in data {
        let out = nn.feedforward(input.view())[[0, 0]];
        sum += out;
        count += 1;
        if out > 0.5 {
            positives += 1;
        }
    }
    let avg = sum / (count as f64);
    println!(
        "Avg output {:.4}, fraction >0.5 {:.2}%, samples {}",
        avg,
        100.0 * (positives as f64) / (count as f64),
        count
    );
}

fn main() {
    env_logger::init();
    let mut n = NeuralNetwork::new(vec![
        default_leaky_relu(2, 8),
        default_leaky_relu(8, 4),
        default_sigmoid(4, 1),
    ]);

    let start = std::time::Instant::now();
    let mut thread_rng = rng();
    let num_epochs: usize = 5;
    let dataset = gen_heart_dataset(-16.0, 16.0, 0.05); // gen_circle_dataset(-2.0, 2.0, 0.01);
    n.train(
        dataset.clone(),
        num_epochs,
        20,
        20,
        0.01,
        0.0,
        &mut thread_rng,
        neural_net::loss::LossFunction::BinaryCrossEntropy,
    );

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

    inspect_predictions(&n, &dataset);

    plot_heatmap(
        &n,
        (-16.0, 16.0),
        (-16.0, 16.0),
        (200, 200),
        "heatmap.png",
        RGBColor(0, 0, 255),
        RGBColor(255, 0, 0),
    )
    .unwrap();

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
