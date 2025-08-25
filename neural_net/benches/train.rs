use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::arr2;
use neural_net::activation::Activation;
use neural_net::layers::Layer;
use neural_net::neural_net::NeuralNetwork;
use rand::rng;
use std::panic;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::AtomicUsize;

fn bench_train(c: &mut Criterion) {
    let mut group = c.benchmark_group("train");
    let panic_count = AtomicUsize::new(0);

    let mut network = NeuralNetwork::new(vec![
        Layer::new(
            arr2(&[[-1.41], [-0.01], [0.41]]),
            arr2(&[[0.0], [0.0], [0.0]]),
            Activation::LeakyReLU,
        ),
        Layer::new(
            arr2(&[
                [0.58, 0.26, -0.44],
                [-0.06, -0.55, -0.09],
                [-0.76, -0.05, -0.64],
            ]),
            arr2(&[[0.0], [0.0], [0.0]]),
            Activation::LeakyReLU,
        ),
        Layer::new(
            arr2(&[[0.47, -0.73, 0.47]]),
            arr2(&[[0.0]]),
            Activation::LeakyReLU,
        ),
    ]);

    let dataset = neural_net::datasets::gen_x2_dataset(0.0, 10.0, 0.1);

    group.bench_function("train_x2", |b| {
        b.iter(|| {
            let result = panic::catch_unwind(AssertUnwindSafe(|| {
                network.train(
                    dataset.clone(),
                    10,
                    10,
                    5,
                    0.001,
                    0.0,
                    &mut rng(),
                    neural_net::loss::LossFunction::BinaryCrossEntropy,
                );
            }));
            if result.is_err() {
                panic_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        });
    });
    println!(
        "Panics: {}",
        panic_count.load(std::sync::atomic::Ordering::Relaxed)
    );
}

criterion_group!(benches, bench_train);
criterion_main!(benches);
