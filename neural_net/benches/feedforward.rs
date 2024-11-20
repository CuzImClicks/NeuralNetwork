use criterion::Criterion;
use neural_net::layers::default_leaky_relu;
use neural_net::neural_net::NeuralNetwork;

fn bench_feedforward(c: &mut Criterion) {
    let mut group = c.benchmark_group("feedforward");
    
    let mut network = NeuralNetwork::new(vec![
        default_leaky_relu(2, 3),
        default_leaky_relu(3, 1),
    ]);
    
    let dataset = neural_net::datasets::gen_xor_dataset();
    
    group.bench_function("feedforward_xor", |b| b.iter(|| {
        for (input, _) in dataset.iter() {
            network.feedforward(input);
        }
    }));
}
