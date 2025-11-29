#[cfg(feature = "gpu")]
use cubecl::prelude::*;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use crate::datasets::Float;
#[cfg(feature = "gpu")]
use crate::{gpu::gpu_tensor::GpuTensor, layers::Layer};

#[derive(Serialize, Deserialize, Debug)]
pub struct TrainingData {
    pub nabla_w: Vec<Array2<Float>>,
    pub nabla_b: Vec<Array2<Float>>,
    pub delta_nabla_w: Vec<Array2<Float>>,
    pub delta_nabla_b: Vec<Array2<Float>>,
    pub activations: Vec<Array2<Float>>,
    pub pre_activations: Vec<Array2<Float>>,
}

#[cfg(feature = "gpu")]
#[derive(Debug)]
pub struct GpuTrainingData<R: Runtime> {
    pub nabla_w: Vec<GpuTensor<R, Float>>,
    pub nabla_b: Vec<GpuTensor<R, Float>>,
    pub delta_nabla_w: Vec<GpuTensor<R, Float>>,
    pub delta_nabla_b: Vec<GpuTensor<R, Float>>,
    pub activations: Vec<GpuTensor<R, Float>>,
    pub pre_activations: Vec<GpuTensor<R, Float>>,
}

#[cfg(feature = "gpu")]
impl<R: Runtime> GpuTrainingData<R> {
    pub fn reset_gradient_buffers(&self, client: &ComputeClient<R::Server>, line_size: u8) {
        for t in self
            .nabla_w
            .iter()
            .chain(&self.nabla_b)
            .chain(&self.delta_nabla_w)
            .chain(&self.delta_nabla_b) {
                launch_reset(client, t, line_size);
            }
    }
    
    pub fn from_layers(layers: &[Layer], client: &ComputeClient<R::Server>) -> Self {
        let nabla_w: Vec<GpuTensor<R, Float>> = layers.iter().map(|l| {
            let shape = l.weights.shape();
            GpuTensor::zeroes(shape.to_vec(), &client)
        }).collect();
        
        let delta_nabla_w: Vec<GpuTensor<R, Float>> = layers.iter().map(|l| {
            let shape = l.weights.shape();
            GpuTensor::zeroes(shape.to_vec(), &client)
        }).collect();
        
        let nabla_b: Vec<GpuTensor<R, Float>> = layers.iter().map(|l| {
            let shape = l.biases.shape();
            GpuTensor::zeroes(shape.to_vec(), &client)
        }).collect();
        
        let delta_nabla_b: Vec<GpuTensor<R, Float>> = layers.iter().map(|l| {
            let shape = l.biases.shape();
            GpuTensor::zeroes(shape.to_vec(), &client)
        }).collect();
        
        let mut activations: Vec<GpuTensor<R, Float>> = layers.iter().map(|l| {
            let dim = l.weights.dim();
            GpuTensor::zeroes(vec![dim.1, 1], &client)
        }).collect();
        
        activations.push(GpuTensor::zeroes(vec![layers.last().unwrap().biases.dim().0, 1], &client));
        
        let mut pre_activations: Vec<GpuTensor<R, Float>> = layers.iter().map(|l| {
            let dim = l.weights.dim();
            GpuTensor::zeroes(vec![dim.1, 1], &client)
        }).collect();
        
        pre_activations.push(GpuTensor::zeroes(vec![layers.last().unwrap().biases.dim().0, 1], &client));
        
        GpuTrainingData {
            nabla_w,
            nabla_b,
            delta_nabla_w,
            delta_nabla_b,
            activations,
            pre_activations,
        }
    }
}


#[inline(always)]
pub fn reset_matrix<O: num_traits::Zero + Copy>(i: &mut [Array2<O>]) {
    for x in i.iter_mut() {
        x.fill(O::zero());
    }
}


#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
pub fn raw_reset_array_gpu(arr: &mut Array<Line<Float>>) {
    

    if ABSOLUTE_POS < arr.len() {
        arr[ABSOLUTE_POS].fill(0.0);
    }
}

#[cfg(feature = "gpu")]
pub fn launch_reset<R: Runtime>(client: &ComputeClient<R::Server>, t: &GpuTensor<R, Float>, line_size: u8) {
    let num_elems: usize = t.shape.iter().product();
    let lines = (num_elems + line_size as usize - 1) / line_size as usize;
    let arg = t.as_array_arg(line_size);
    unsafe {
        raw_reset_array_gpu::launch_unchecked(&client, CubeCount::Static(1, 1, 1), CubeDim::new(lines as u32, 1, 1), arg);
    }
}

#[cfg(feature = "gpu")]
pub fn reset_all<'a, R, I>(
    client: &ComputeClient<R::Server>,
    tensors: I,
    line_size: u8,
) where
    R: Runtime,
    I: IntoIterator<Item = &'a GpuTensor<R, Float>>,
{
    for t in tensors {
        launch_reset(client, t, line_size);
    }
}

#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
pub fn raw_accumulate_and_reset(nabla: &mut Array<Line<Float>>, delta: &mut Array<Line<Float>>) {
    if ABSOLUTE_POS < nabla.len() {
        nabla[ABSOLUTE_POS] = nabla[ABSOLUTE_POS] + delta[ABSOLUTE_POS];
        delta[ABSOLUTE_POS].fill(0.0);
    }
}

#[cfg(feature = "gpu")]
pub fn launch_accumulate_and_reset<R: Runtime>(client: &ComputeClient<R::Server>,
    nabla: &GpuTensor<R, Float>,
    delta: &GpuTensor<R, Float>,
    line_size: u8) {
    debug_assert_eq!(nabla.shape, delta.shape);
    
    let num_elems: usize = nabla.shape.iter().product();
    let lines = (num_elems + line_size as usize - 1) / line_size as usize;
    
    let arg_nabla = nabla.as_array_arg(line_size);
    let arg_delta = delta.as_array_arg(line_size);
    
    unsafe {
        raw_accumulate_and_reset::launch_unchecked(client, CubeCount::Static(1, 1, 1), CubeDim::new(lines as u32, 1, 1), arg_nabla, arg_delta);
    }
}