use crate::{
    LINE_SIZE, datasets::Float, gpu::{gpu_operations::{add_inplace, apply_activation_function_inplace, matmul}, gpu_tensor::GpuTensor}, layers::{GpuLayer, Layer}, loss::LossFunction, metadata::Metadata, saving_and_loading::{Format, save_to_file}, training_data::TrainingData, training_events::{Callbacks, EpochStats, TrainingEvent}
};
use crate::training_data::{launch_accumulate_and_reset, reset_matrix};
use anyhow::{Context, Result};
use cubecl::{prelude::*, server::Allocation, std::tensor::TensorHandle};
use log::info;
use ndarray::{Array2, ArrayBase, ArrayView, ArrayView2, Data, Dimension, Ix2, RawData};
use rand::{Rng, prelude::SliceRandom};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{fmt::Display, mem, path::Path, process::exit, time::Instant};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
    pub metadata: Option<Metadata>,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>) -> Self {
        NeuralNetwork {
            layers,
            metadata: None,
        }
    }

    pub fn with_metadata(&mut self, metadata: Metadata) {
        self.metadata = Some(metadata)
    }

    pub fn feedforward(&self, inputs: ArrayView2<Float>) -> Array2<Float> {
        let mut result = inputs.to_owned();
        for layer in self.layers.iter() {
            result = layer.weights.dot(&result);
            result += &layer.biases;
            result.mapv_inplace(|v| layer.activation.apply(v));
        }
        result
    }

    pub fn validate(
        &self,
        data: &[(ArrayView2<Float>, ArrayView2<Float>)],
        loss: &LossFunction,
    ) -> Float {
        let (sum, count) = data
            .par_iter()
            .map(|(input, truth)| {
                let output = self.feedforward(*input);
                loss.apply(output.view(), *truth)
            })
            .fold(|| (0.0 as Float, 0usize), |(acc, cnt), l| (acc + l, cnt + 1))
            .reduce(|| (0.0 as Float, 0usize), |(a1, c1), (a2, c2)| (a1 + a2, c1 + c2));
        if count == 0 {
            0.0
        } else {
            sum / (count as Float)
        }
    }

    pub fn train(
        &mut self,
        raw_training_data: Vec<(Array2<Float>, Array2<Float>)>,
        epochs: usize,
        batches_per_epoch: usize,
        batch_size: usize,
        learning_rate: Float,
        lambda: Float,
        rng: &mut impl Rng,
        loss_function: LossFunction,
        mut callbacks: Callbacks,
    ) {
        let training_start = Instant::now();
        callbacks.on_event(
            self,
            TrainingEvent::TrainingBegin {
                start_time: training_start,
                total_epochs: epochs,
            },
        );

        let weight_size: Vec<Array2<Float>> = self
            .layers
            .iter()
            .map(|w| Array2::zeros(w.weights.dim()))
            .collect();
        let bias_size: Vec<Array2<Float>> = self
            .layers
            .iter()
            .map(|w| Array2::zeros(w.biases.dim()))
            .collect();
        let mut activations_shape: Vec<Array2<Float>> = weight_size
            .iter()
            .map(|w| Array2::zeros((w.dim().1, 1)))
            .collect();

        activations_shape.push(Array2::zeros((
            self.layers.last().unwrap().biases.dim().0,
            1,
        )));

        let mut training_data = TrainingData {
            nabla_w: weight_size.clone(),
            nabla_b: bias_size.clone(),
            delta_nabla_w: weight_size,
            delta_nabla_b: bias_size,
            activations: activations_shape.clone(),
            pre_activations: activations_shape,
        };

        let mut data = raw_training_data;
        data.shuffle(rng);

        #[cfg(feature = "loss")]
        let validation_pairs = {
            let val_size = ((data.len() as Float * 0.1).ceil() as usize).clamp(1, 100);
            data.drain(..val_size).collect::<Vec<_>>()
        };
        #[cfg(feature = "loss")]
        let views = validation_pairs
            .iter()
            .map(|(i, o)| (i.view(), o.view()))
            .collect::<Vec<(ArrayView2<Float>, ArrayView2<Float>)>>();

        let mut indices: Vec<usize> = (0..data.len()).collect::<Vec<usize>>();

        for epoch in 0..epochs {
            let epoch_start = Instant::now();
            indices.shuffle(rng);
            let shuffle_elapsed = epoch_start.elapsed();

            for batch in indices.chunks(batch_size).take(batches_per_epoch) {
                self.update_weights_biases(
                    &batch.iter().map(|it| &data[*it]).collect::<Vec<_>>(),
                    learning_rate,
                    lambda,
                    &mut training_data,
                    &loss_function,
                );
            }

            let backpropagation_elapsed = epoch_start.elapsed() - shuffle_elapsed;
            #[cfg(not(feature = "loss"))]
            {
                callbacks.on_event(TrainingEvent::EpochEnd {
                    stats: EpochStats {
                        epoch,
                        duration: epoch_start.elapsed(),
                        .._
                    },
                })
            }
            #[cfg(feature = "loss")]
            {
                let loss = self.validate(&views, &loss_function);
                callbacks.on_event(
                    self,
                    TrainingEvent::EpochEnd {
                        stats: EpochStats {
                            epoch,
                            loss,
                            loss_elapsed: epoch_start.elapsed()
                                - (backpropagation_elapsed + shuffle_elapsed),
                            backpropagation_elapsed: backpropagation_elapsed,
                            shuffle_elapsed: shuffle_elapsed,
                            total_elapsed: epoch_start.elapsed(),
                        },
                    },
                );
            }
        }

        callbacks.on_event(
            self,
            TrainingEvent::TrainingEnd {
                end_time: training_start.elapsed(),
                total_epochs: epochs,
                training_dataset: data,
                validation_dataset: validation_pairs,
            },
        );
    }

    fn update_weights_biases(
        &mut self,
        batch: &[&(Array2<Float>, Array2<Float>)],
        learning_rate: Float,
        lambda: Float,
        training_data: &mut TrainingData,
        loss: &LossFunction,
    ) {
        let nabla_w = &mut training_data.nabla_w;
        let nabla_b = &mut training_data.nabla_b;
        let delta_nabla_w = &mut training_data.delta_nabla_w;
        let delta_nabla_b = &mut training_data.delta_nabla_b;
        let activations = &mut training_data.activations;
        let pre_activations = &mut training_data.pre_activations;
        reset_matrix(nabla_w);
        reset_matrix(nabla_b);
        reset_matrix(delta_nabla_w);
        reset_matrix(delta_nabla_b);
        for (i, t) in batch {
            self.backpropagation(
                i,
                t,
                delta_nabla_w,
                delta_nabla_b,
                activations,
                pre_activations,
                loss,
            );
            for (nw, dnw) in nabla_w.iter_mut().zip(delta_nabla_w.iter()) {
                *nw += dnw;
            }
            for (nb, dnb) in nabla_b.iter_mut().zip(delta_nabla_b.iter()) {
                *nb += dnb;
            }
            reset_matrix(delta_nabla_w);
            reset_matrix(delta_nabla_b);
        }

        let batch_size_f = batch.len() as Float;
        for (layer, grad_w) in self.layers.iter_mut().zip(nabla_w.iter()) {
            if lambda != 0.0 {
                // decouple decay: w := w * (1 - lr * lambda / batch_size)
                layer
                    .weights
                    .mapv_inplace(|w| w * (1.0 - learning_rate * lambda / batch_size_f));
            }
            layer
                .weights
                .scaled_add(-learning_rate / batch_size_f, grad_w);
        }

        for (layer, grad_b) in self.layers.iter_mut().zip(nabla_b.iter()) {
            layer
                .biases
                .scaled_add(-learning_rate / batch_size_f, grad_b);
        }
    }

    #[allow(non_snake_case)]
    fn backpropagation(
        &mut self,
        input: &Array2<Float>,
        truth: &Array2<Float>,
        delta_nabla_w: &mut [Array2<Float>],
        delta_nabla_b: &mut [Array2<Float>],
        activations: &mut [Array2<Float>],
        pre_activations: &mut [Array2<Float>],
        loss: &LossFunction,
    ) {
        let num_layers = self.layers.len();

        activations[0].assign(input);
        for (i, layer) in self.layers.iter().enumerate() {
            let mut z = layer.weights.dot(&activations[i]);
            z += &layer.biases;
            pre_activations[i + 1].assign(&z);

            let mut a = z.clone();
            a.mapv_inplace(|v| layer.activation.apply(v));
            activations[i + 1].assign(&a);
        }

        let a_L = activations[num_layers].view();
        let z_L = pre_activations[num_layers].view();
        let mut delta = loss.derivative(a_L, truth.view()); // dL/da

        if matches!(loss, LossFunction::MeanErrorSquared) {
            let output_activation = &self.layers[num_layers - 1].activation;
            delta.zip_mut_with(&z_L, |d, &z| {
                *d *= output_activation.derivative(z);
            });
        }

        delta_nabla_b[num_layers - 1].assign(&delta);
        delta_nabla_w[num_layers - 1].assign(&delta.dot(&activations[num_layers - 1].t()));

        for l in (0..num_layers - 1).rev() {
            // backpropagate through weights
            delta = self.layers[l + 1].weights.t().dot(&delta);

            let z_l = pre_activations[l + 1].view();
            let activation_l = &self.layers[l].activation;
            delta.zip_mut_with(&z_l, |d, &z| {
                *d *= activation_l.derivative(z);
            });

            delta_nabla_b[l].assign(&delta);
            delta_nabla_w[l].assign(&delta.dot(&activations[l].t()));
        }
    }

    pub fn save_checkpoint(&self, path: impl AsRef<Path>, format: Format) -> Result<()> {
        save_to_file(path, self, format)
    }

    #[cfg(feature = "gpu")]
    pub fn train_gpu<R: Runtime>(
        &mut self,
        raw_training_data: Vec<(Array2<Float>, Array2<Float>)>,
        epochs: usize,
        batches_per_epoch: usize,
        batch_size: usize,
        learning_rate: Float,
        lambda: Float,
        rng: &mut impl Rng,
        loss_function: LossFunction,
        mut callbacks: Callbacks,
        device: &R::Device,
    ) -> Result<()> {
        use crate::{gpu::gpu_tensor::GpuTensor, training_data::GpuTrainingData};

        let client = R::client(&device);

        let mut data = raw_training_data;
        data.shuffle(rng);

        #[cfg(feature = "loss")]
        let validation_pairs = {
            let val_size = ((data.len() as Float * 0.1).ceil() as usize).max(1);
            data.drain(..val_size).collect::<Vec<_>>()
        };
        #[cfg(feature = "loss")]
        let views = validation_pairs
            .iter()
            .map(|(i, o)| (i.view(), o.view()))
            .collect::<Vec<(ArrayView2<Float>, ArrayView2<Float>)>>();

        let gpu_data: Vec<(GpuTensor<R, Float>, GpuTensor<R, Float>)> = data.iter().map(|(i, o)| {
            (GpuTensor::<R, Float>::copy(i, &client), GpuTensor::<R, Float>::copy(o, &client))
        }).collect();

        let gpu_layers = self.get_weights_as_tensors::<R>(&device)?;
    
        let mut training_data = GpuTrainingData::<R>::from_layers(&self.layers, &client);
        
        let mut indices: Vec<usize> = (0..data.len()).collect::<Vec<usize>>();

        for epoch in 0..epochs {
            let epoch_start = Instant::now();
            indices.shuffle(rng);
            let shuffle_elapsed = epoch_start.elapsed();

            for batch in indices.chunks(batch_size).take(batches_per_epoch) {
                
                
                
                for (nw, dnw) in training_data.nabla_w.iter().zip(training_data.delta_nabla_w.iter()) {
                    launch_accumulate_and_reset(&client, nw, dnw, LINE_SIZE);
                }
                
                for (nb, dnb) in training_data.nabla_b.iter().zip(training_data.delta_nabla_b.iter()) {
                    launch_accumulate_and_reset(&client, nb, dnb, LINE_SIZE);
                }    
            }
        }

        Ok(())
    }
    
    pub fn feedforward_gpu<R: Runtime>(&self, input: GpuTensor<R, Float>, device: &R::Device) -> Result<GpuTensor<R, Float>> {
        let client = R::client(&device);
        let num_layers = self.layers.len();
        
        let gpu_layers: Vec<GpuLayer<R>> = self.get_weights_as_tensors(device)?;
        
        let mut result: Vec<GpuTensor<R, Float>> = Vec::with_capacity(num_layers);
        result.push(input);
        
        for (i, layer) in &mut gpu_layers.iter().enumerate() {
            let z = matmul(&layer.weights, &result[i], &client, LINE_SIZE);
            add_inplace(&z, &layer.biases, &client, LINE_SIZE);
            apply_activation_function_inplace(&z, &layer.activation, &client, LINE_SIZE);
            result.push(z);
        }
        
        
        Ok(result.last().unwrap().clone())
    }

    #[cfg(feature = "gpu")]
    pub fn get_weights_as_tensors<R: Runtime>(&self, device: &R::Device) -> Result<Vec<GpuLayer<R>>> {
        let client = R::client(&device);
        let mut gpu_layers: Vec<GpuLayer<R>> = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            let weights = GpuTensor::<R, Float>::copy(&layer.weights, &client);

            let biases = GpuTensor::<R, Float>::copy(&layer.biases, &client);
            
            gpu_layers.push(GpuLayer {
                weights,
                biases,
                activation: layer.activation.clone(),
            });
        }

        Ok(gpu_layers)
    }
}

pub fn print_matrix<T: Display>(matrix: &ArrayView<T, Ix2>) {
    for row in matrix.rows() {
        for cell in row {
            print!("{cell:.4}, ");
        }
        println!();
    }
}

#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
pub fn raw_forward_pass(activation: Array<Line<Float>>, weights: Array<Line<Float>>) {
    
}

pub fn launch_forward_pass<R: Runtime>(activation_buf: GpuTensor<R, Float>) {
    
}
