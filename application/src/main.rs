use std::process::exit;
use ndarray::{arr2, Array2};
use num_traits::Pow;
use neural_net::datasets;
use neural_net::layers::{default_leaky_relu, default_linear, default_sigmoid, default_tanh};
use neural_net::neural_net::{print_matrix, NeuralNetwork};


fn gen_heart_dataset(low: f64, high: f64, step: f64) -> Vec<(Array2<f64>, Array2<f64>)> {
    let mut dataset = Vec::new();

    let mut x  = low;
    let mut y: f64;
    // x=16sin^3(x) y=13cos(x)-5cos(2x)-2cos(3x)-cos(4x)
    while x < high {
        y = low;

        let s: f64 = x.signum() * (x / 16.0).abs().pow(1.0/3.0);
        let A: f64 = -6.0 + 18.0 * s.powi(2) - 8.0 * s.pow(4.0);
        let B: f64 = (11.0 + 8.0 * s.pow(2.0)) * (1.0 - s.powi(2)).sqrt();
        let upper = A + B;
        let lower = A - B;
        while y < high {
            if y <= upper && y >= lower {
                dataset.push((
                    arr2(&[[x], [y]]),
                    arr2(&[[1_f64]])
                    )
                );
                //print!("X")
            } else {
                dataset.push((
                    arr2(&[[x], [y]]),
                    arr2(&[[0_f64]])
                )
                );
                //print!("_")
            }
            y += step;
        }
        x += step;
        println!();
    }
    
    dataset


}

fn main() {
    let mut n = NeuralNetwork::new(vec![
        default_sigmoid(2, 6),
        default_sigmoid(6, 6),
        default_sigmoid(6, 6),
        default_linear(6, 1),
    ]);

    let start = std::time::Instant::now();

    let num_epochs: usize = 2308;
    let dataset = gen_heart_dataset(-16.0, 16.0, 0.1); // datasets::gen_circle_dataset(0.0, 5.0, 0.01);
    n.train(dataset, num_epochs, 20, 30, 0.001, 0.0);

    let elapsed = start.elapsed();
    println!("Elapsed: {:?}", elapsed);
    println!("{:.2?} epochs/s", (num_epochs as f32) / elapsed.as_secs_f32());

    println!("\nWeights:");
    for (i, l) in n.layers.iter().enumerate() {
        println!("{}", i);
        print_matrix(&l.weights.view());
    }

    println!("\nBiases: ");
    for (i, l) in n.layers.iter().enumerate() {
        println!("{}", i);
        print_matrix(&l.biases.view());
    }

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