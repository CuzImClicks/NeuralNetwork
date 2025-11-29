
// CubeCL Book

use std::{marker::PhantomData};

use cubecl::{prelude::*, server::Handle, std::tensor::compact_strides};
use ndarray::{ArrayBase, Data, Dimension};

/// Simple GpuTensor
#[derive(Debug)]
pub struct GpuTensor<R: Runtime, F: Float + CubeElement> {
    pub data: Handle,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    _r: PhantomData<R>,
    _f: PhantomData<F>,
}

impl<R: Runtime, F: Float + CubeElement> Clone for GpuTensor<R, F> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            _r: PhantomData,
            _f: PhantomData,
        }
    }
}

impl<R: Runtime, F: Float + CubeElement> GpuTensor<R, F> {
    /// Create an empty GpuTensor with a shape
    pub fn empty(shape: Vec<usize>, client: &ComputeClient<R::Server>) -> Self {
        let size = shape.iter().product::<usize>() * core::mem::size_of::<F>();
        let data = client.empty(size);

        let strides = compact_strides(&shape);
        Self {
            data,
            shape,
            strides,
            _r: PhantomData,
            _f: PhantomData,
        }
    }
    
    pub fn copy<T: Data<Elem = F>, D: Dimension>(arr: &ArrayBase<T, D>, client: &ComputeClient<R::Server>) -> Self {
        assert!(
            arr.is_standard_layout(),
            "gpu_allocate_vec: array must be standard-layout (contiguous). \
             Call .to_owned() on a copy if needed."
        );
        
        let slice = arr.as_slice().unwrap();
        let bytes = F::as_bytes(slice);
        let shape = arr.shape();
        let allocation = client.create_from_slice(
            bytes,
        );
        
        Self { data: allocation, shape: shape.to_vec(), strides: compact_strides(shape), _r: PhantomData, _f: PhantomData }
    }
    
    pub fn zeroes(shape: Vec<usize>, client: &ComputeClient<R::Server>) -> Self {

        let size: usize = shape.iter().product();
        let vec: Vec<F>  = vec![F::from(0.0).unwrap(); size];
        let bytes = F::as_bytes(&vec);
        let allocation = client.create_from_slice(
            bytes,
        );
        let strides = compact_strides(&shape);
        
        Self { data: allocation, shape, strides, _r: PhantomData, _f: PhantomData }
    }
    
    pub fn from_slice(shape: Vec<usize>, data: &[F], client: &ComputeClient<R::Server>) -> Self {
        let bytes = F::as_bytes(data);
        let allocation = client.create_from_slice(
            bytes,
        );
        
        Self { data: allocation, shape: shape.to_vec(), strides: compact_strides(&shape), _r: PhantomData, _f: PhantomData }
    }

    /// Create a TensorArg to pass to a kernel
    pub fn as_tensor_arg(&self, line_size: u8) -> TensorArg<'_, R> {
        unsafe { TensorArg::from_raw_parts::<F>(&self.data, &self.strides, &self.shape, line_size) }
    }
    
    pub fn as_array_arg(&self, line_size: u8) -> ArrayArg<'_, R> {
        unsafe { ArrayArg::from_raw_parts::<F>(&self.data, self.shape.iter().product(), line_size) }
    }

    /// Return the data from the client
    pub fn read(self, client: &ComputeClient<R::Server>) -> Vec<F> {
        let bytes = client.read_one(self.data);
        F::from_bytes(&bytes).to_vec()
    }
}