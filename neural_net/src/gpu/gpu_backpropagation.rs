
#[cfg(feature = "gpu")]
use cubecl::prelude::*;

#[cfg(feature = "gpu")]
use crate::{LINE_SIZE, datasets::Float, gpu::{gpu_operations::{add_inplace, apply_activation_function_inplace, copy_into, matmul_into}, gpu_tensor::GpuTensor}, layers::GpuLayer, neural_net::NeuralNetwork, training_data::GpuTrainingData};

#[cfg(feature = "gpu")]
pub(crate) fn backpropagation_gpu<R: Runtime>(gpu_layers: &[GpuLayer<R>], input: &GpuTensor<R, Float>, training_data: &mut GpuTrainingData<R>, client: &ComputeClient<R::Server>) {
    training_data.activations[0] = input.clone();
    for (i, layer) in gpu_layers.iter().enumerate() {
        matmul_into(&layer.weights, &training_data.activations[i], &training_data.pre_activations[i+1], &client, LINE_SIZE);
        add_inplace(&training_data.pre_activations[i+1], &layer.biases,  &client, LINE_SIZE);
        copy_into(&training_data.activations[i+1], &training_data.pre_activations[i+1], &client, LINE_SIZE);
        apply_activation_function_inplace(&training_data.activations[i+1], &layer.activation, &client, LINE_SIZE);
    }
    
    let last_activation = &training_data.activations[gpu_layers.len()];
    let last_pre_activation = &training_data.pre_activations[gpu_layers.len()];
    //let mut delta = apply 
    // TODO
}