use std::f64::consts::E;
use std::ops::Div;

struct Neuron {
    weights: Vec<f64>,
    bias: f64
}

impl Neuron {
    fn output(&self, inputs: &[f64]) -> f64 {
        self.activate(dot(&self.weights, inputs)) + self.bias
    }
    fn activate(&self, output: f64) -> f64 {
        output
    }
}

struct Layer {
    neurons: Vec<Neuron>
}

impl Layer {
    fn propagate(&self, inputs: &[f64]) -> Vec<f64> {
        self.neurons.iter().map(|neuron|neuron.output(inputs)).collect()
    }
}

struct NeuralNetwork {
    layers: Vec<Layer>
}

impl NeuralNetwork {
    pub fn evaluate(&self, inputs: &[f64]) -> Vec<f64> {
        self.layers.iter().fold(inputs.to_vec(), |acc, layer| layer.propagate(&acc))
    }
    pub fn train(&mut self, training_data: Vec<(Vec<f64>, Vec<f64>)>, epochs: usize, learning_rate: f64, loss_function: fn(&[f64], &[f64]) -> f64) {
        for _ in 0..epochs {
            for (inputs, expected) in training_data.iter() {
                let outputs = self.evaluate(inputs);

                let loss = loss_function(&outputs, expected);
                println!("{}", loss);
            }
        }
    }
    fn backpropagate(&self, predicted: &[f64], actual: &[f64]) {

    }

    fn gradients(predicted: &[f64], actual: &[f64], fun: fn((&f64, &f64)) -> f64) -> Vec<f64> {
        predicted.iter().zip(actual).map(fun).collect()
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

fn sigmoid_derivative(x: f64) -> f64 {
    x * (1.0 - x)
}

fn dot(one: &[f64], two: &[f64]) -> f64 {
    one.iter().zip(two).map(|(x, y)| x * y).sum()
}


fn mean_squared_error(output: &[f64], predicted: &[f64]) -> f64 {
    return output.iter()
        .zip(predicted)
        .map(|(act, pred)| (pred - act).powi(2) )
        .sum::<f64>().div(output.len() as f64);
}


fn main() {
    let first_layer = Layer {
        neurons: vec![
            Neuron { weights: vec![0.15, 0.20], bias: 0.35 },
            Neuron { weights: vec![0.25, 0.30], bias: 0.35 },
        ],
    };

    let second_layer = Layer {
        neurons: vec![
            Neuron { weights: vec![0.40, 0.45], bias: 0.60 },
            Neuron { weights: vec![0.50, 0.55], bias: 0.60 },
        ],
    };
    let mut neural_net = NeuralNetwork { layers: vec![first_layer, second_layer]};
    //println!("{:?}", neural_net.evaluate(&[1_f64, 2_f64, 3_f64]));
    neural_net.train(vec![(vec![1_f64, 2_f64, 3_f64], vec![1.6_f64, 1.9_f64])], 10, 0.3, mean_squared_error)
}
