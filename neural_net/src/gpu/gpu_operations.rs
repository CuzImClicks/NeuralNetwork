#[cfg(feature = "gpu")]
use cubecl::prelude::*;

use crate::{datasets::Float};
#[cfg(feature = "gpu")]
use crate::{LINE_SIZE, activation::Activation, gpu::gpu_tensor::GpuTensor};

// a = [M * K]
// b = [K, N]
// idx = ABSOLUTE_POS
// row = idx / N
// col = idx % N
#[cfg(feature = "gpu")]
#[allow(non_snake_case)]
#[cube(launch_unchecked)]
pub fn raw_matmul(a: Array<Float>, b: Array<Float>, c: &mut Array<Float>, M: u32, K: u32, N: u32) {
    if ABSOLUTE_POS < M * N {
        let row = ABSOLUTE_POS / N;
        let col = ABSOLUTE_POS % N;
        let c_idx = row * N + col;
        let mut acc: Float = 0.0;
        for i in 0..K {
            acc += a[row * K + i] * b[i * N + col];
        }
        c[c_idx] = acc;
    }
}

#[cfg(feature = "gpu")]
pub fn matmul<R: Runtime>(a: &GpuTensor<R, Float>, b: &GpuTensor<R, Float>, client: &ComputeClient<R::Server>, line_size: u8) -> GpuTensor<R, Float> {
    let M = a.shape[0] as u32;
    let K = a.shape[1] as u32;
    
    let N = b.shape[1] as u32;
    let output_buf = GpuTensor::<R, Float>::empty(vec![M as usize, N as usize], client);
    
    let num_elems: usize = output_buf.shape.iter().product::<usize>();
    let lines = (num_elems + line_size as usize - 1) / line_size as usize;
    let a_arg = a.as_array_arg(1);
    let b_arg = b.as_array_arg(1);
    
    unsafe {
        raw_matmul::launch_unchecked(client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(lines as u32, 1, 1),
        a_arg, b_arg, 
        output_buf.as_array_arg(line_size),
        ScalarArg::new(M), ScalarArg::new(K), ScalarArg::new(N));
    }
    
    output_buf
}

#[cfg(feature = "gpu")]
pub fn matmul_into<R: Runtime>(a: &GpuTensor<R, Float>,b: &GpuTensor<R, Float>, output_buf: &GpuTensor<R, Float>, client: &ComputeClient<R::Server>, line_size: u8) {
    let M = a.shape[0] as u32;
    let K = a.shape[1] as u32;
    
    let N = b.shape[1] as u32;
    
    let num_elems: usize = output_buf.shape.iter().product::<usize>();
    let lines = (num_elems + line_size as usize - 1) / line_size as usize;
    let a_arg = a.as_array_arg(1);
    let b_arg = b.as_array_arg(1);
    
    unsafe {
        raw_matmul::launch_unchecked(client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(lines as u32, 1, 1),
        a_arg, b_arg,
        output_buf.as_array_arg(line_size),
        ScalarArg::new(M), ScalarArg::new(K), ScalarArg::new(N));
    }
}

#[cfg(feature = "gpu")]
#[allow(non_snake_case)]
#[cube(launch_unchecked)]
pub fn raw_elementwise_add(a: Array<Float>, b: Array<Float>, c: &mut Array<Float>, M: u32, N: u32) {
    if ABSOLUTE_POS < c.len() {
        c[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}

#[cfg(feature = "gpu")]
pub fn add_elementwise<R: Runtime>(a: &GpuTensor<R, Float>, b: &GpuTensor<R, Float>, client: &ComputeClient<R::Server>, line_size: u8) -> GpuTensor<R, Float> {
    let M = a.shape[0] as u32;
    let N = a.shape[1] as u32;
    
    let output_buf = GpuTensor::<R, Float>::empty(vec![M as usize, N as usize], client);
    
    let num_elems: usize = output_buf.shape.iter().product::<usize>();
    let lines = (num_elems + line_size as usize - 1) / line_size as usize;
    let a_arg = a.as_array_arg(1);
    let b_arg = b.as_array_arg(1);
    
    unsafe {
        raw_elementwise_add::launch_unchecked(client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(lines as u32, 1, 1),
        a_arg, b_arg, 
        output_buf.as_array_arg(line_size),
        ScalarArg::new(M),
        ScalarArg::new(N));
    }
    
    output_buf
}

#[cfg(feature = "gpu")]
#[allow(non_snake_case)]
#[cube(launch_unchecked)]
pub fn raw_add_inplace(a: &mut Array<Float>, b: Array<Float>, N: u32) {
    if ABSOLUTE_POS < a.len() {
        let row = ABSOLUTE_POS / N;
        a[ABSOLUTE_POS] += b[row];
    }
}

#[cfg(feature = "gpu")]
pub fn add_inplace<R: Runtime>(a: &GpuTensor<R, Float>, b: &GpuTensor<R, Float>, client: &ComputeClient<R::Server>, line_size: u8) {
    let N = a.shape[1] as u32;
    
    let num_elems: usize = a.shape.iter().product::<usize>();
    let lines = (num_elems + line_size as usize - 1) / line_size as usize;
    let a_arg = a.as_array_arg(1);
    let b_arg = b.as_array_arg(1);
    
    unsafe {
        raw_add_inplace::launch_unchecked(client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(lines as u32, 1, 1),
        a_arg, b_arg,
        ScalarArg::new(N));
    }
}

#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
pub fn raw_sigmoid(a: &mut Array<Float>) {

    if ABSOLUTE_POS >= a.len() { terminate!(); };
    a[ABSOLUTE_POS] = 1.0 / (1.0 + Float::exp(-a[ABSOLUTE_POS]))
}

#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
pub fn raw_relu(a: &mut Array<Float>) {

    if ABSOLUTE_POS >= a.len() { terminate!(); };
    if a[ABSOLUTE_POS] <= 0.0 { 
        a[ABSOLUTE_POS] = 0.0;
    }
}

#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
pub fn raw_leaky_relu(a: &mut Array<Float>) {

    if ABSOLUTE_POS >= a.len() { terminate!(); };
    if a[ABSOLUTE_POS] <= 0.0 { 
        a[ABSOLUTE_POS] = a[ABSOLUTE_POS] * 0.01;
    }
}

#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
pub fn raw_tanh(a: &mut Array<Float>) {

    if ABSOLUTE_POS >= a.len() { terminate!(); };
    a[ABSOLUTE_POS] = Float::tanh(a[ABSOLUTE_POS])
}

#[cfg(feature = "gpu")]
pub fn apply_activation_function_inplace<R: Runtime>(a: &GpuTensor<R, Float>, activation: &Activation, client: &ComputeClient<R::Server>, line_size: u8) {

    let num_elems: usize = a.shape.iter().product::<usize>();
    let lines = (num_elems + line_size as usize - 1) / line_size as usize;
    
    match activation {
        Activation::Sigmoid => unsafe {
            raw_sigmoid::launch_unchecked(client, CubeCount::Static(1, 1, 1), CubeDim::new(lines as u32, 1, 1), a.as_array_arg(LINE_SIZE))
        },
        Activation::ReLU =>  unsafe {
            raw_relu::launch_unchecked(client, CubeCount::Static(1, 1, 1), CubeDim::new(lines as u32, 1, 1), a.as_array_arg(LINE_SIZE))
        },
        Activation::LeakyReLU =>  unsafe {
            raw_leaky_relu::launch_unchecked(client, CubeCount::Static(1, 1, 1), CubeDim::new(lines as u32, 1, 1), a.as_array_arg(LINE_SIZE))
        },
        Activation::Tanh =>  unsafe {
            raw_tanh::launch_unchecked(client, CubeCount::Static(1, 1, 1), CubeDim::new(lines as u32, 1, 1), a.as_array_arg(LINE_SIZE))
        },
        Activation::Linear => {},
    }
}

/// 

#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
pub fn raw_sigmoid_derivative(a: &mut Array<Float>) {

    if ABSOLUTE_POS >= a.len() { terminate!(); };
    let s = 1.0 / (1.0 + Float::exp(-a[ABSOLUTE_POS]));
    a[ABSOLUTE_POS] = s * (1.0 -s)
}

#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
pub fn raw_relu_derivative(a: &mut Array<Float>) {

    if ABSOLUTE_POS >= a.len() { terminate!(); };
    if a[ABSOLUTE_POS] > 0.0 { 
        a[ABSOLUTE_POS] = 1.0;
    } else {
        a[ABSOLUTE_POS] = 0.0
    }
}

#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
pub fn raw_leaky_relu_derivative(a: &mut Array<Float>) {

    if ABSOLUTE_POS >= a.len() { terminate!(); };
    if a[ABSOLUTE_POS] > 0.0 { 
        a[ABSOLUTE_POS] = 1.0;
    } else {
        a[ABSOLUTE_POS] = 0.01
    }
}

#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
pub fn raw_tanh_derivative(a: &mut Array<Float>) {

    if ABSOLUTE_POS >= a.len() { terminate!(); };
    let s = Float::tanh(a[ABSOLUTE_POS]);
    a[ABSOLUTE_POS] = s * -s + 1.0
}

#[cfg(feature = "gpu")]
pub fn apply_activation_function_derivative_inplace<R: Runtime>(a: &GpuTensor<R, Float>, activation: Activation, client: &ComputeClient<R::Server>, line_size: u8) {
    let num_elems: usize = a.shape.iter().product::<usize>();
    let lines = (num_elems + line_size as usize - 1) / line_size as usize;
    
    match activation {
        Activation::Sigmoid => unsafe {
            raw_sigmoid_derivative::launch_unchecked(client, CubeCount::Static(1, 1, 1), CubeDim::new(lines as u32, 1, 1), a.as_array_arg(LINE_SIZE))
        },
        Activation::ReLU =>  unsafe {
            raw_relu_derivative::launch_unchecked(client, CubeCount::Static(1, 1, 1), CubeDim::new(lines as u32, 1, 1), a.as_array_arg(LINE_SIZE))
        },
        Activation::LeakyReLU =>  unsafe {
            raw_leaky_relu_derivative::launch_unchecked(client, CubeCount::Static(1, 1, 1), CubeDim::new(lines as u32, 1, 1), a.as_array_arg(LINE_SIZE))
        },
        Activation::Tanh =>  unsafe {
            raw_tanh_derivative::launch_unchecked(client, CubeCount::Static(1, 1, 1), CubeDim::new(lines as u32, 1, 1), a.as_array_arg(LINE_SIZE))
        },
        Activation::Linear => {},
    }
}

pub const E_FLOAT: Float = {
    #[cfg(feature = "f32")]
    { std::f32::consts::E }

    #[cfg(not(feature = "f32"))]
    { std::f64::consts::E }
};

//#[cfg(feature = "gpu")]
//#[cube(launch_unchecked)]
//pub fn raw_binary_cross_entropy_inplace(output: &mut Array<Float>, truth: Array<Float>) {
//
//    if ABSOLUTE_POS < output.len() {
//
//        let eps = 1e-12;
//        let o = output[ABSOLUTE_POS].max(eps).min(1.0-eps);
//        output[ABSOLUTE_POS] = -(truth[ABSOLUTE_POS] * (o.log(E_FLOAT)) + &(1.0 - truth[ABSOLUTE_POS]) * (1.0 - o));
//    };
//}

//pub fn apply_loss_inplace<R: Runtime>(a: &GpuTensor<R, Float>, loss: LossFunction, client: &ComputeClient<R::Server>, line_size: u8) {
//    let num_elems: usize = a.shape.iter().product::<usize>();
//    let lines = (num_elems + line_size as usize - 1) / line_size as usize;
//    
//    match loss {
//        LossFunction::MeanErrorSquared => todo!(),
//        LossFunction::BinaryCrossEntropy => todo!(),
//    }
//}


/// Copies `b` into `a`
#[cfg(feature = "gpu")]
#[cube( launch_unchecked)]
pub fn raw_copy_into(a: &mut Array<Line<Float>>, b: Array<Line<Float>>) {
    if ABSOLUTE_POS < a.len() {
        a[ABSOLUTE_POS] = b[ABSOLUTE_POS]
    }
}

#[cfg(feature = "gpu")]
pub fn copy_into<R: Runtime>(a: &GpuTensor<R, Float>, b: &GpuTensor<R, Float>, client: &ComputeClient<R::Server>, line_size: u8) {
    let num_elems: usize = a.shape.iter().product::<usize>();
    let lines = (num_elems + line_size as usize - 1) / line_size as usize;
    
    unsafe {
        raw_copy_into::launch_unchecked(client, CubeCount::Static(1, 1, 1), CubeDim::new(lines as u32, 1, 1), a.as_array_arg(line_size), b.as_array_arg(line_size));
    }
}