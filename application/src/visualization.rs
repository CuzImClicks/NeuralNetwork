use std::{f64, path::Path, vec};

use anyhow::{Result, anyhow};
use ndarray::{Axis, array};
use neural_net::{datasets::Float, neural_net::NeuralNetwork};
use plotters::prelude::*;

/// Linear interpolation between two RGBColor endpoints.
fn interpolate_color(low: &RGBColor, high: &RGBColor, t: Float) -> RGBColor {
    let clamp = |v: Float| {
        if v < 0.0 {
            0
        } else if v > 255.0 {
            255
        } else {
            v as u8
        }
    };
    let r = clamp((1.0 - t) * (low.0 as Float) + t * (high.0 as Float));
    let g = clamp((1.0 - t) * (low.1 as Float) + t * (high.1 as Float));
    let b = clamp((1.0 - t) * (low.2 as Float) + t * (high.2 as Float));
    RGBColor(r, g, b)
}

pub fn plot_heatmap<P: AsRef<Path>>(
    nn: &NeuralNetwork,
    x_range: (Float, Float),
    y_range: (Float, Float),
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

    let mut values: Vec<Float> = vec![0.0; nx * ny];
    let mut min_v = Float::INFINITY;
    let mut max_v = Float::NEG_INFINITY;

    for ix in 0..nx {
        let x = xmin + (xmax - xmin) * (ix as Float) / ((nx - 1) as Float);
        for iy in 0..ny {
            let y = ymin + (ymax - ymin) * (iy as Float) / ((ny - 1) as Float);
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

    let range = (max_v - min_v).max(Float::EPSILON);

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

    let dx = (xmax - xmin) / ((nx - 1) as Float);
    let dy = (ymax - ymin) / ((ny - 1) as Float);

    for ix in 0..nx {
        let x0 = xmin + (xmax - xmin) * (ix as Float) / ((nx - 1) as Float);
        for iy in 0..ny {
            let y0 = ymin + (ymax - ymin) * (iy as Float) / ((ny - 1) as Float);
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
    x_range: (Float, Float),
    y_range: (Float, Float),
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

    let mut values: Vec<[Float; 3]> = vec![[0.0; 3]; nx * ny];

    for ix in 0..nx {
        let x = xmin + (xmax - xmin) * (ix as Float) / ((nx - 1) as Float);
        for iy in 0..ny {
            let y = ymin + (ymax - ymin) * (iy as Float) / ((ny - 1) as Float);
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

    let dx = (xmax - xmin) / ((nx - 1) as Float);
    let dy = (ymax - ymin) / ((ny - 1) as Float);

    for ix in 0..nx {
        let x0 = xmin + (xmax - xmin) * (ix as Float) / ((nx - 1) as Float);
        for iy in 0..ny {
            let y0 = ymin + (ymax - ymin) * (iy as Float) / ((ny - 1) as Float);
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
    data: Vec<(Float, Float)>,
    caption: &str,
    filename: &str,
    resolution: (u32, u32),
    x_range: (Float, Float),
    y_range: (Float, Float),
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

pub fn plot_points<P: AsRef<Path>>(
    data: Vec<(f64, f64)>,
    caption: &str,
    path: P,
    resolution: (u32, u32),
    x_range: (f64, f64),
    y_range: (f64, f64),
) -> Result<()> {
    let root = BitMapBackend::new(path.as_ref(), resolution).into_drawing_area();
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

    //chart.draw_series(PointSeries::new(data, 2.0, &BLUE))?;

    root.present()?;
    Ok(())
}

fn grayscale_between_bounds(low: Float, high: Float, value: Float) -> RGBColor {
    debug_assert!(value >= low);
    let c = 255.0 * ((value - low) / (high - low)).powf(0.7).clamp(0.1, 0.9);
    RGBColor(c as u8, c as u8, c as u8)
}

pub fn visualize_neural_network<P: AsRef<Path>>(
    nn: &NeuralNetwork,
    path: P,
    size: (u32, u32),
) -> Result<()> {
    let root = SVGBackend::new(path.as_ref(), size).into_drawing_area();
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
    let width = width as i32;
    let height = height as i32;

    let left_pad: i32 = 100;
    let right_pad: i32 = 100;
    let top_pad: i32 = 60;
    let bottom_pad: i32 = 60;

    let usable_w = width - left_pad - right_pad;
    let usable_h = height - top_pad - bottom_pad;
    let radius: i32 = 10;

    let layer_count = nn.layers.len();

    let x_step = if layer_count > 1 {
        usable_w / layer_count as i32
    } else {
        0
    };

    let padding = 20;
    let y_step = 2 * radius + padding;

    let input_layer = &nn.layers[0];

    let mut x: i32 = left_pad + radius;
    let mut y: i32 = top_pad
        + radius
        + (usable_h - radius - (input_layer.weights.len_of(Axis(1)) as i32 * y_step)) / 2;

    // input layer
    for _ in &mut input_layer.weights.axis_iter(Axis(1)) {
        let circle = &Circle::new((x, y), radius, ShapeStyle::from(&BLACK).stroke_width(1));
        root.draw(circle)?;

        y += y_step;
    }

    x += x_step;

    let mut last_layer_y = top_pad
        + radius
        + (usable_h - radius - (input_layer.weights.len_of(Axis(1)) as i32 * y_step)) / 2;

    for layer in &nn.layers {
        y = top_pad
            + radius
            + (usable_h - radius - (layer.weights.len_of(Axis(0)) as i32 * y_step)) / 2;
        let biases = layer
            .biases
            .axis_iter(Axis(0))
            .map(|it| it[0])
            .collect::<Vec<Float>>();
        for (j, neuron) in &mut layer.weights.axis_iter(Axis(0)).enumerate() {
            let bias = biases[j];
            let circle = &Circle::new(
                (x, y),
                radius,
                ShapeStyle::from(grayscale_between_bounds(min_bias, max_bias, bias))
                    .stroke_width(1),
            );
            root.draw(circle)?;
            for (i, weight) in neuron.iter().enumerate() {
                root.draw(&PathElement::new(
                    vec![
                        (
                            ((x - x_step + radius)),
                            (last_layer_y + (y_step * i as i32)),
                        ),
                        (((x - radius)), (y)),
                    ],
                    ShapeStyle::from(grayscale_between_bounds(min_weight, max_weight, *weight)),
                ))?;
            }
            y += y_step;
        }
        last_layer_y = top_pad
            + radius
            + (usable_h - radius - (layer.weights.len_of(Axis(0)) as i32 * y_step)) / 2;
        x += x_step;
    }

    root.present()?;
    Ok(())
}

pub fn weight_histogram<P: AsRef<Path>>(nn: &NeuralNetwork, path: P) -> Result<()> {
    let root = BitMapBackend::new(path.as_ref(), (600, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut max_weight = 0.0;
    let mut min_weight = 0.0;

    nn.layers.iter().for_each(|it| {
        it.weights.for_each(|weight| {
            if *weight < min_weight {
                min_weight = *weight;
            }
            if *weight > max_weight {
                max_weight = *weight;
            }
        });
    });

    let mut flattened: Vec<_> = nn
        .layers
        .iter()
        .flat_map(|it| it.weights.flatten())
        .collect();

    flattened.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let x_step = (max_weight - min_weight) * 0.05;

    let mut amounts = [0; 20];
    flattened.iter().for_each(|it| {
        (0..20).for_each(|i| {
            if min_weight + i as Float * x_step <= *it && min_weight + (i + 1) as Float * x_step > *it {
                amounts[i] += 1;
            }
        })
    });

    let mut chart = ChartBuilder::on(&root)
        .caption("Weight Distribution", ("sans-serif", 24))
        .margin(10)
        .x_label_area_size(35)
        .y_label_area_size(35)
        .build_cartesian_2d(
            min_weight..max_weight,
            0_usize..(*amounts.iter().max().unwrap() + 1),
        )?;

    chart.configure_mesh().draw()?;

    chart.draw_series(amounts.iter().enumerate().map(|(i, v)| {
        let x0 = min_weight + i as Float * x_step;
        let x1 = (min_weight + (i + 1) as Float * x_step).min(max_weight);
        let y0 = 0;
        let y1 = v;
        
        Rectangle::new([(x0, y0), (x1, *y1)], RED.filled())
    }))?;

    Ok(())
}
