use ndarray::array;
use neural_net::neural_net::NeuralNetwork;
use plotters::prelude::*;
use std::error::Error;

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

/// Draws a heatmap of `nn`'s scalar output over the given x/y range. The lowest network
/// output becomes `low_color`, highest becomes `high_color`. Saves to `filename`.
pub fn plot_heatmap(
    nn: &NeuralNetwork,
    x_range: (f64, f64),
    y_range: (f64, f64),
    resolution: (usize, usize), // number of cells along x and y
    filename: &str,
    low_color: RGBColor,
    high_color: RGBColor,
) -> Result<(), Box<dyn Error>> {
    let (xmin, xmax) = x_range;
    let (ymin, ymax) = y_range;
    assert!(xmax > xmin && ymax > ymin, "Invalid ranges");
    let (nx, ny) = resolution;
    assert!(
        nx >= 2 && ny >= 2,
        "Resolution must be at least 2 in each axis"
    );

    // Sample the network over the grid and track min/max values.
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

    // avoid degenerate range
    let range = (max_v - min_v).max(std::f64::EPSILON);

    // Prepare drawing area with extra width for colorbar
    let image_size = (1000, 800); // width x height
    let root = BitMapBackend::new(filename, image_size).into_drawing_area();
    root.fill(&WHITE)?;

    // Split left/right: 80% heatmap, 20% legend
    let (heatmap_area, legend_area) = root.split_horizontally((80).percent_width());

    // Build heatmap chart
    let mut chart = ChartBuilder::on(&heatmap_area)
        .margin(10)
        .caption("Network Output Heatmap", ("sans-serif", 18).into_font())
        .x_label_area_size(35)
        .y_label_area_size(35)
        .build_cartesian_2d(xmin..xmax, ymin..ymax)?;
    chart.configure_mesh().draw()?;

    // cell sizes in data coordinates
    let dx = (xmax - xmin) / ((nx - 1) as f64);
    let dy = (ymax - ymin) / ((ny - 1) as f64);

    for ix in 0..nx {
        let x0 = xmin + (xmax - xmin) * (ix as f64) / ((nx - 1) as f64);
        for iy in 0..ny {
            let y0 = ymin + (ymax - ymin) * (iy as f64) / ((ny - 1) as f64);
            let v = values[ix + iy * nx];
            let t = (v - min_v) / range; // normalized to [0,1]
            let color = interpolate_color(&low_color, &high_color, t);
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x0, y0), (x0 + dx, y0 + dy)],
                ShapeStyle::from(&color).filled(),
            )))?;
        }
    }

    Ok(())
}
