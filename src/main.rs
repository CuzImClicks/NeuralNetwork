mod activation;

use std::fmt::{Debug, write};
use std::ops::Div;
use crate::activation::{sigmoid, sigmoid_derivative};

struct Neuron {
    weights: Vec<f64>,
    bias: f64,
    pre_activation_output: f64,
    inputs: Vec<f64>
}

impl Neuron {
    fn new(weights: Vec<f64>, bias: f64) -> Self {
        Neuron {
            weights,
            bias,
            pre_activation_output: 0.0,
            inputs: Vec::new()
        }
    }
    fn output(&mut self, inputs: &[f64], activation_function: fn(f64) -> f64) -> f64 {
        self.inputs = inputs.to_vec();
        self.pre_activation_output = dot(&self.weights, inputs) + self.bias;
        activation_function(self.pre_activation_output)
    }
}

struct Layer {
    neurons: Vec<Neuron>
}

impl Layer {
    fn evaluate(&mut self, inputs: &[f64], activation_function: fn(f64) -> f64) -> Vec<f64> {
        self.neurons.iter_mut().map(|neuron|neuron.output(inputs, activation_function)).collect()
    }
}

struct NeuralNetwork {
    layers: Vec<Layer>,
    activation_function: fn(f64) -> f64
}

impl NeuralNetwork {
    pub fn propagate(&mut self, inputs: &[f64]) -> Vec<f64> {
        self.layers.iter_mut().fold(inputs.to_vec(), |acc, layer| layer.evaluate(&acc, self.activation_function))
    }

    pub fn train(&mut self, training_data: Vec<(Vec<f64>, Vec<f64>)>,
                 epochs: usize,
                 learning_rate: f64) {
        self.fit(training_data, epochs, learning_rate, mean_squared_error, sigmoid_derivative)
    }

    pub fn fit(&mut self,
               training_data: Vec<(Vec<f64>, Vec<f64>)>,
               epochs: usize,
               learning_rate: f64,
               loss_function: fn(&[f64], &[f64]) -> f64,
               activation_function_derivative: fn(f64) -> f64
    ) {

        for epoch in 0..epochs {
            //println!("Epoch: {}", epoch+1);
            //println!("1 0 -> {:?}", self.propagate(&[1f64, 0f64]));
            //println!("0 1 -> {:?}", self.propagate(&[0f64, 1f64]));
            //println!("0 0 -> {:?}", self.propagate(&[0f64, 0f64]));
            //println!("1 1 -> {:?}", self.propagate(&[1f64, 1f64]));
            for (step, (inputs, expected)) in training_data.iter().enumerate() {
                let outputs = self.propagate(inputs);

                let loss = loss_function(&outputs, expected);
                 //println!("Step {} - Loss: {}", step, loss);
                self.backpropagate(inputs, &outputs, expected, learning_rate, activation_function_derivative);
            }

        }

    }

    fn backpropagate(&mut self, inputs: &[f64], outputs: &[f64], expected: &[f64], learning_rate: f64, activation_function_derivative: fn(f64) -> f64) {

    }
}


fn dot(one: &[f64], two: &[f64]) -> f64 {
    one.iter().zip(two).map(|(x, y)| x * y).sum()
}


fn mean_squared_error(output: &[f64], predicted: &[f64]) -> f64 {
    return output.iter()
        .zip(predicted)
        .map(|(pred, target)| (target - pred).powi(2) )
        .sum::<f64>().div(output.len() as f64);
}


fn main() {
    let first_layer = Layer {
        neurons: vec![
            Neuron::new(vec![0.15, -0.25], 0.0),
            Neuron::new(vec![0.15, -0.25], 0.0),
        ],
    };

    let second_layer = Layer {
        neurons: vec![
            Neuron::new(vec![0.35, -0.35], 0.0),
        ],
    };
    let mut neural_net = NeuralNetwork { layers: vec![first_layer, second_layer], activation_function: sigmoid};
    println!("{:?}", neural_net.propagate(&[1f64, 0f64]));
    neural_net.fit(vec![(vec![0f64, 0f64], vec![0f64]), (vec![1f64, 0f64], vec![1f64]), (vec![0f64, 1f64], vec![1f64]), (vec![1f64, 1f64], vec![0f64])], 1000, 1f64, mean_squared_error, sigmoid_derivative);

    //print out the layers and the weights and biases of the neurons
    for layer in neural_net.layers.iter() {
        for neuron in layer.neurons.iter() {
            print!("({:?}, {}) ", neuron.weights, neuron.bias);
        }
        println!()
    }
    println!("1 0 -> {:?}", neural_net.propagate(&[1f64, 0f64]));
    println!("0 1 -> {:?}", neural_net.propagate(&[0f64, 1f64]));
    println!("0 0 -> {:?}", neural_net.propagate(&[0f64, 0f64]));
    println!("1 1 -> {:?}", neural_net.propagate(&[1f64, 1f64]));

}
