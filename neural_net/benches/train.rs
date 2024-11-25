use std::panic;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::AtomicUsize;
use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::arr2;
use neural_net::activation::{leaky_relu, leaky_relu_prime};
use neural_net::layers::Layer;
use neural_net::neural_net::NeuralNetwork;

fn bench_train(c: &mut Criterion) {
    let mut group = c.benchmark_group("train");
    let panic_count = AtomicUsize::new(0);
    
    let mut network = NeuralNetwork::new(vec![
        Layer::new(arr2(&[[-1.41], [-0.01], [0.41]]), arr2(&[[0.0], [0.0], [0.0]]), leaky_relu, leaky_relu_prime),
        Layer::new(arr2(&[[0.58, 0.26, -0.44], [-0.06, -0.55, -0.09], [-0.76, -0.05, -0.64]]), arr2(&[[0.0], [0.0], [0.0]]), leaky_relu, leaky_relu_prime),
        Layer::new(arr2(&[[0.47, -0.73, 0.47]]), arr2(&[[0.0]]), leaky_relu, leaky_relu_prime),
    ]);

    let dataset = neural_net::datasets::gen_x2_dataset(0.0, 10.0, 0.1);

    group.bench_function("train_x2", |b| {
        b.iter(|| {
            let result = panic::catch_unwind(AssertUnwindSafe(|| {
                network.train(dataset.clone(), 10, 10, 5, 0.001, 0.0);
            }));
            if result.is_err() {
                panic_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        });
    });
    println!("Panics: {}", panic_count.load(std::sync::atomic::Ordering::Relaxed));
}

criterion_group!(benches, bench_train);
criterion_main!(benches);