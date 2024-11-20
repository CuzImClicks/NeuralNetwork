use neural_net::datasets;
use neural_net::layers::default_leaky_relu;
use neural_net::neural_net::{print_matrix, NeuralNetwork};

fn main() {
    let mut n = NeuralNetwork::new(vec![
        default_leaky_relu(1, 3),
        default_leaky_relu(3, 1),
    ]);

    let start = std::time::Instant::now();

    let num_epochs: usize = 10000;
    let dataset = datasets::gen_x2_dataset(0.0, 10.0, 0.1); // datasets::gen_circle_dataset(0.0, 5.0, 0.01);
    n.train(dataset.clone(), num_epochs, 10, 10, 0.01, 0.0);

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
    //    println!("{} -> {}", i, output[[0, 0]]);
    //    i += 0.1;
    //}
}