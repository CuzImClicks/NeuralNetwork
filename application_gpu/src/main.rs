use anyhow::Result;
use cubecl::{cuda::{CudaDevice, CudaRuntime}, prelude::*, wgpu::{WgpuDevice, WgpuRuntime}};
use log::LevelFilter;
use neural_net::{checkpoints::CheckpointStrategy, datasets::{Float, gen_heart_dataset}, gpu::{gpu_operations::matmul, gpu_tensor::GpuTensor}, layers::{default_leaky_relu, default_sigmoid}, neural_net::{NeuralNetwork, print_matrix}, saving_and_loading::load_from_file, training_events::{Callbacks, Logger, LossCollector}};
use rand::rng;
use std::io::Write;
use neural_net::loss::LossFunction::BinaryCrossEntropy;


fn main() -> Result<()>{
    env_logger::builder()
        .filter_level(LevelFilter::Info)
        .parse_default_env()
        .format_timestamp(Some(env_logger::TimestampPrecision::Millis))
        .format(|buf, record| {
            let style = buf.default_level_style(record.level());
            let timestamp = buf.timestamp_seconds();
            writeln!(
                buf,
                "{timestamp} [{style}{}{style:#}] [{}] {}",
                record.level(),
                record.module_path().unwrap(),
                record.args(),
            )
        })
        .init();
    
    
    let mut n = load_from_file("./", neural_net::saving_and_loading::Format::Json).unwrap_or_else(|_| {
        NeuralNetwork::new(vec![
            default_leaky_relu(2, 16),
            default_leaky_relu(16, 8),
            default_leaky_relu(8, 16),
            default_leaky_relu(16, 1),
            default_sigmoid(1, 1)
        ])
    });
    
    let num_epochs = 1;
    let data = gen_heart_dataset(-16.0, 16.0, 0.1);
    
    //let mut loss_collector = LossCollector::new(num_epochs);
    //let mut logger = Logger {};
    //let mut checkpoint_strategy = CheckpointStrategy::Percentage {
    //    percentage: 0.1,
    //    total_epochs: num_epochs,
    //    folder: "./heart2/checkpoints",
    //};
    
    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);
    let output_cpu = n.feedforward(data[0].0.view());
    let output_gpu = n.feedforward_gpu(GpuTensor::<WgpuRuntime, Float>::copy::<_, _>(&data[0].0, &client), &device)?;
    
    print_matrix(&output_cpu.view());
    println!("{:?}", output_gpu.read(&client));
    
    //n.train_gpu::<WgpuRuntime>(data,
    //    num_epochs,
    //    40,
    //    60,
    //    0.001,
    //    0.0,
    //    &mut rng(),
    //    BinaryCrossEntropy,
    //    Callbacks::new(vec![
    //        &mut logger,
    //        &mut loss_collector,
    //        &mut checkpoint_strategy,
    //    ]),
    //    &device)?;
    
    Ok(())
}
