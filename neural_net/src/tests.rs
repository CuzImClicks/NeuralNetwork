#[cfg(test)]
mod tests {
    use crate::{
        datasets::{Float, gen_xor_dataset},
        layers::{default_leaky_relu, default_sigmoid},
        loss::LossFunction,
        neural_net::NeuralNetwork,
        training_events::Callbacks,
    };
    use ndarray::{Array2, ArrayView2, array};
    use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};

    /// Exhaustive 3-bit parity dataset: input shape (3,1), output shape (1,1)
    fn gen_parity3_dataset() -> Vec<(Array2<Float>, Array2<Float>)> {
        let mut data = Vec::with_capacity(8);
        for i in 0..8 {
            let b0 = ((i >> 0) & 1) as Float;
            let b1 = ((i >> 1) & 1) as Float;
            let b2 = ((i >> 2) & 1) as Float;
            let input = array![[b0], [b1], [b2]]; // shape (3,1)
            let parity = ((b0 as i32 + b1 as i32 + b2 as i32) % 2) as Float;
            let truth = array![[parity]]; // shape (1,1)
            data.push((input, truth));
        }
        data
    }

    #[test]
    fn overfit_3bit_parity() {
        let mut rng = ChaCha20Rng::seed_from_u64(0xDEADBEEF);

        let mut nn = NeuralNetwork::new(vec![
            default_leaky_relu(3, 8),
            default_leaky_relu(8, 4),
            default_sigmoid(4, 1),
        ]);

        let dataset = gen_parity3_dataset();

        let train_views: Vec<(ndarray::ArrayView2<Float>, ndarray::ArrayView2<Float>)> =
            dataset.iter().map(|(i, o)| (i.view(), o.view())).collect();

        nn.train(
            dataset.clone(),
            5000,
            1,
            8,
            0.1,
            0.0,
            &mut rng,
            LossFunction::BinaryCrossEntropy,
            Callbacks::default(),
        );

        let training_loss = nn.validate(&train_views, &LossFunction::BinaryCrossEntropy);
        assert!(
            training_loss < 0.01,
            "Expected near-zero training loss, got {training_loss}"
        );

        for (input, truth) in dataset {
            let output = nn.feedforward(input.view())[[0, 0]];
            let pred = if output > 0.5 { 1.0 } else { 0.0 };
            assert_eq!(
                pred,
                truth[[0, 0]],
                "Expected {} got {pred} for input {input:?} (raw output {output:.5})",
                truth[[0, 0]],
            );
        }
    }

    #[test]
    fn xor() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let mut net = NeuralNetwork::new(vec![default_sigmoid(2, 2), default_sigmoid(2, 1)]);

        let data = gen_xor_dataset();

        net.train(
            data.clone(),
            5000,
            4,
            1,
            0.1,
            0.0,
            &mut rng,
            crate::loss::LossFunction::BinaryCrossEntropy,
            Callbacks::default(),
        );

        let loss = net.validate(
            &data
                .iter()
                .map(|(i, o)| (i.view(), o.view()))
                .collect::<Vec<(ArrayView2<Float>, ArrayView2<Float>)>>(),
            &crate::loss::LossFunction::BinaryCrossEntropy,
        );
        assert!(loss < 0.01, "Expected near-zero training loss, got {loss}");

        for (input, truth) in data {
            let output = net.feedforward(input.view());
            let out_val = output[[0, 0]];
            let predicted = if out_val > 0.5 { 1.0 } else { 0.0 };
            let expected = truth[[0, 0]];
            assert!(
                (predicted - expected).abs() < 1e-8,
                "XOR failed for input {input:?}: expected {expected}, got {out_val:.4} (raw), predicted {predicted}",
            );
        }
    }
}
