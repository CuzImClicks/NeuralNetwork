use ndarray::ArrayView2;
use rand::prelude::{Rng, SliceRandom};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::{datasets::gen_xor_dataset, layers::default_sigmoid, neural_net::NeuralNetwork};

#[test]
fn xor() {
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let mut net = NeuralNetwork::new(vec![default_sigmoid(2, 2), default_sigmoid(2, 1)]);

    let data = gen_xor_dataset();

    net.train(data.clone(), 50000, 4, 1, 0.5, 0.0, &mut rng);

    let loss = net.validate(
        &data
            .iter()
            .map(|(i, o)| (i.view(), o.view()))
            .collect::<Vec<(ArrayView2<f64>, ArrayView2<f64>)>>(),
    );
    assert!(
        loss < 0.05,
        "Expected low MSE after training XOR, got loss={:.6}",
        loss
    );

    // Check discrete outputs (threshold at 0.5)
    for (input, truth) in data {
        let output = net.feedforward(input.view());
        let out_val = output[[0, 0]];
        let predicted = if out_val > 0.5 { 1.0 } else { 0.0 };
        let expected = truth[[0, 0]];
        assert!(
            (predicted - expected).abs() < 1e-8,
            "XOR failed for input {:?}: expected {}, got {:.4} (raw), predicted {}",
            input,
            expected,
            out_val,
            predicted
        );
    }
}
