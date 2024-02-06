mod activation;

use std::ops::Div;
use crate::activation::sigmoid;

struct Neuron {
    weights: Vec<f64>,
    bias: f64,
    pre_activation_output: f64
}

impl Neuron {
    fn new(weights: Vec<f64>, bias: f64) -> Self {
        Neuron {
            weights,
            bias,
            pre_activation_output: 0.0
        }
    }
    fn output(&mut self, inputs: &[f64], activation_function: fn(f64) -> f64) -> f64 {
        self.pre_activation_output = dot(&self.weights, inputs) + self.bias;
        activation_function(self.pre_activation_output)
    }
}

struct Layer {
    neurons: Vec<Neuron>
}

impl Layer {
    fn propagate(&mut self, inputs: &[f64], activation_function: fn(f64) -> f64) -> Vec<f64> {
        self.neurons.iter_mut().map(|mut neuron|neuron.output(inputs, activation_function)).collect()
    }
}

struct NeuralNetwork {
    layers: Vec<Layer>,
    activation_function: fn(f64) -> f64
}

impl NeuralNetwork {
    pub fn evaluate(&mut self, inputs: &[f64]) -> Vec<f64> {
        self.layers.iter_mut().fold(inputs.to_vec(), |acc, layer| layer.propagate(&acc, self.activation_function))
    }
    pub fn train(&mut self,
                 training_data: Vec<(Vec<f64>, Vec<f64>)>,
                 epochs: usize,
                 learning_rate: f64,
                 loss_function: fn(&[f64], &[f64]) -> f64,
                 activation_function_derivative: fn(f64) -> f64
    ) {

        for _ in 0..epochs {
            for (inputs, expected) in training_data.iter() {
                let outputs = self.evaluate(inputs);

                let loss = loss_function(&outputs, expected);
                self.backpropagate(inputs, &outputs, expected, activation_function_derivative);
            }
        }

    }
    fn backpropagate(&mut self, inputs: &[f64], outputs: &[f64], expected: &[f64], activation_function_derivative: fn(f64) -> f64) {
        let mut deltas: Vec<f64> = Vec::new();

        for (output, &actual) in outputs.iter().zip(expected) {
            let delta = (output - actual) * activation_function_derivative(*output);
            deltas.push(delta);
        }

        let current_inputs = outputs;
        for (i, layer) in self.layers.iter().enumerate().rev() {
            let new_deltas = layer.neurons
        }

    }
}


fn dot(one: &[f64], two: &[f64]) -> f64 {
    one.iter().zip(two).map(|(x, y)| x * y).sum()
}


fn mean_squared_error(output: &[f64], predicted: &[f64]) -> f64 {
    return output.iter()
        .zip(predicted)
        .map(|(act, pred)| (act - pred).powi(2) )
        .sum::<f64>().div(output.len() as f64);
}


fn main() {
    let first_layer = Layer {
        neurons: vec![
            Neuron::new(vec![0.15, 0.20], 0.35),
            Neuron::new(vec![0.25, 0.30], 0.35),
        ],
    };

    let second_layer = Layer {
        neurons: vec![
            Neuron::new(vec![0.40, 0.45], 0.60),
            Neuron::new(vec![0.50, 0.55], 0.60),
        ],
    };
    let mut neural_net = NeuralNetwork { layers: vec![first_layer, second_layer], activation_function: sigmoid};
    //println!("{:?}", neural_net.evaluate(&[1_f64, 2_f64, 3_f64]));
    //neural_net.train(vec![(vec![1_f64, 2_f64, 3_f64], vec![1.6_f64, 1.9_f64])], 10, 0.3, mean_squared_error, sigmoid)
}
