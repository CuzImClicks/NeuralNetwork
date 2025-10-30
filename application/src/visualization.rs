use std::{f64, fs, path::Path};

use anyhow::{Result, anyhow};
use ndarray::{Axis, array};
use neural_net::neural_net::NeuralNetwork;
use plotters::prelude::*;

/// Linear interpolation between two RGBColor endpoints.
fn interpolate_color(low: &RGBColor, high: &RGBColor, t: f64) -> RGBColor {
    let clamp = |v: f64| {
        if v < 0.0 {
            0
        } else if v > 255.0 {
            255
        } else {
            v as u8
        }
    };
    let r = clamp((1.0 - t) * (low.0 as f64) + t * (high.0 as f64));
    let g = clamp((1.0 - t) * (low.1 as f64) + t * (high.1 as f64));
    let b = clamp((1.0 - t) * (low.2 as f64) + t * (high.2 as f64));
    RGBColor(r, g, b)
}

pub fn plot_heatmap<P: AsRef<Path>>(
    nn: &NeuralNetwork,
    x_range: (f64, f64),
    y_range: (f64, f64),
    resolution: (usize, usize),
    file: P,
    low_color: RGBColor,
    high_color: RGBColor,
) -> Result<()> {
    let (xmin, xmax) = x_range;
    let (ymin, ymax) = y_range;
    assert!(xmax > xmin && ymax > ymin, "Invalid ranges");
    let (nx, ny) = resolution;
    assert!(
        nx >= 2 && ny >= 2,
        "Resolution must be at least 2 in each axis"
    );

    let mut values = vec![0.0f64; nx * ny];
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;

    for ix in 0..nx {
        let x = xmin + (xmax - xmin) * (ix as f64) / ((nx - 1) as f64);
        for iy in 0..ny {
            let y = ymin + (ymax - ymin) * (iy as f64) / ((ny - 1) as f64);
            let input = array![[x], [y]];
            let output = nn.feedforward(input.view());
            let value = output[[0, 0]];
            values[ix + iy * nx] = value;
            if value < min_v {
                min_v = value;
            }
            if value > max_v {
                max_v = value;
            }
        }
    }

    let range = (max_v - min_v).max(std::f64::EPSILON);

    let image_size = (1000, 800);
    let root = BitMapBackend::new(&file, image_size).into_drawing_area();
    root.fill(&WHITE)?;

    let (heatmap_area, _) = root.split_horizontally((80).percent_width());

    let mut chart = ChartBuilder::on(&heatmap_area)
        .margin(10)
        .caption("Network Output Heatmap", ("sans-serif", 18).into_font())
        .x_label_area_size(35)
        .y_label_area_size(35)
        .build_cartesian_2d(xmin..xmax, ymin..ymax)?;
    chart.configure_mesh().draw()?;

    let dx = (xmax - xmin) / ((nx - 1) as f64);
    let dy = (ymax - ymin) / ((ny - 1) as f64);

    for ix in 0..nx {
        let x0 = xmin + (xmax - xmin) * (ix as f64) / ((nx - 1) as f64);
        for iy in 0..ny {
            let y0 = ymin + (ymax - ymin) * (iy as f64) / ((ny - 1) as f64);
            let v = values[ix + iy * nx];
            let t = (v - min_v) / range;
            let color = interpolate_color(&low_color, &high_color, t);
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x0, y0), (x0 + dx, y0 + dy)],
                ShapeStyle::from(&color).filled(),
            )))?;
        }
    }

    Ok(())
}

pub fn plot_rgb(
    nn: &NeuralNetwork,
    x_range: (f64, f64),
    y_range: (f64, f64),
    resolution: (usize, usize),
    filename: &str,
) -> Result<()> {
    let (xmin, xmax) = x_range;
    let (ymin, ymax) = y_range;
    assert!(xmax > xmin && ymax > ymin, "Invalid ranges");
    let (nx, ny) = resolution;
    assert!(
        nx >= 2 && ny >= 2,
        "Resolution must be at least 2 in each axis"
    );

    let mut values = vec![[0.0_f64; 3]; nx * ny];

    for ix in 0..nx {
        let x = xmin + (xmax - xmin) * (ix as f64) / ((nx - 1) as f64);
        for iy in 0..ny {
            let y = ymin + (ymax - ymin) * (iy as f64) / ((ny - 1) as f64);
            let input = array![[x], [y]];
            let output = nn.feedforward(input.view());
            let value = output.axis_iter(Axis(1)).collect::<Vec<_>>();
            let slice = value[0]
                .to_slice()
                .ok_or_else(|| anyhow!("Could not convert to slice"))?;

            values[ix + iy * nx] = slice.try_into()?;
        }
    }

    let image_size = (1000, 800);
    let root = BitMapBackend::new(filename, image_size).into_drawing_area();
    root.fill(&WHITE)?;

    let (heatmap_area, _) = root.split_horizontally((80).percent_width());

    let mut chart = ChartBuilder::on(&heatmap_area)
        .margin(10)
        .caption("Network Output Heatmap", ("sans-serif", 18).into_font())
        .x_label_area_size(35)
        .y_label_area_size(35)
        .build_cartesian_2d(xmin..xmax, ymin..ymax)?;
    chart.configure_mesh().draw()?;

    let dx = (xmax - xmin) / ((nx - 1) as f64);
    let dy = (ymax - ymin) / ((ny - 1) as f64);

    for ix in 0..nx {
        let x0 = xmin + (xmax - xmin) * (ix as f64) / ((nx - 1) as f64);
        for iy in 0..ny {
            let y0 = ymin + (ymax - ymin) * (iy as f64) / ((ny - 1) as f64);
            let v = values[ix + iy * nx];

            let r = (v[0] * 255.0).round().clamp(0.0, 255.0) as u8;
            let g = (v[1] * 255.0).round().clamp(0.0, 255.0) as u8;
            let b = (v[2] * 255.0).round().clamp(0.0, 255.0) as u8;
            let color = RGBColor(r, g, b);
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x0, y0), (x0 + dx, y0 + dy)],
                ShapeStyle::from(&color).filled(),
            )))?;
        }
    }

    Ok(())
}

pub fn plot_line(
    data: Vec<(f64, f64)>,
    caption: &str,
    filename: &str,
    resolution: (u32, u32),
    x_range: (f64, f64),
    y_range: (f64, f64),
) -> Result<()> {
    let root = BitMapBackend::new(filename, resolution).into_drawing_area();
    root.fill(&WHITE)?;

    let (x_min, x_max) = x_range;
    let (y_min, y_max) = y_range;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 24))
        .margin(10)
        .x_label_area_size(35)
        .y_label_area_size(35)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(data, &BLUE))?;

    root.present()?;
    Ok(())
}

pub fn visualize_neural_network(nn: &NeuralNetwork, filename: &str) -> Result<()> {
    let root = BitMapBackend::new(filename, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut max_weight = 0.0;
    let mut min_weight = 0.0;

    let mut max_bias = 0.0;
    let mut min_bias = 0.0;

    nn.layers.iter().for_each(|it| {
        it.weights.for_each(|weight| {
            if *weight < min_weight {
                min_weight = *weight;
            }
            if *weight > max_weight {
                max_weight = *weight;
            }
        });
        it.biases.for_each(|bias| {
            if *bias < min_bias {
                min_bias = *bias;
            }
            if *bias > max_bias {
                max_bias = *bias;
            }
        });
    });

    let (width, height) = root.dim_in_pixel();

    let left_pad = 100i32;
    let right_pad = 100i32;
    let top_pad = 60i32;
    let bottom_pad = 60i32;

    let usable_w = (width as i32) - left_pad - right_pad;
    let usable_h = (height as i32) - top_pad - bottom_pad;

    let layer_count = nn.layers.len();

    let x_step = if layer_count > 1 {
        usable_w as f64 / (layer_count as f64 - 1.0)
    } else {
        0.0
    };

    root.present()?;
    Ok(())
}
