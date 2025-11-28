
use std::f32::consts::TAU;

use ndarray::{Array2, arr2};

#[cfg(feature = "f64")]
pub type Float = f64;

#[cfg(feature = "f32")]
pub type Float = f32;

#[cfg(feature = "f16")]
pub type Float = f16;

pub fn gen_x2_dataset(low: Float, high: Float, step: Float) -> Vec<(Array2<Float>, Array2<Float>)> {
    let mut v: Vec<(Array2<Float>, Array2<Float>)> = Vec::with_capacity((high / step) as usize);
    let mut i = low;
    while i <= high {
        v.push((arr2(&[[i]]), arr2(&[[i.powi(2)]])));
        i += step;
    }
    v
}

pub fn gen_xor_dataset() -> Vec<(Array2<Float>, Array2<Float>)> {
    vec![
        (arr2(&[[0.0], [0.0]]), arr2(&[[0.0]])),
        (arr2(&[[0.0], [1.0]]), arr2(&[[1.0]])),
        (arr2(&[[1.0], [0.0]]), arr2(&[[1.0]])),
        (arr2(&[[1.0], [1.0]]), arr2(&[[0.0]])),
    ]
}

pub fn gen_circle_dataset(low: Float, high: Float, step: Float) -> Vec<(Array2<Float>, Array2<Float>)> {
    let mut dataset = Vec::new();

    let mut x = low;
    while x <= high {
        let mut y = low;
        while y <= high {
            let distance = ((x - 2.0).powi(2) + (y - 2.0).powi(2)).sqrt();

            let label = if distance < 1.0 { 1.0 } else { 0.0 };

            dataset.push((arr2(&[[x], [y]]), arr2(&[[label]])));

            y += step;
        }
        x += step;
    }

    dataset
}

#[allow(non_snake_case)]
pub fn gen_heart_dataset(low: Float, high: Float, step: Float) -> Vec<(Array2<Float>, Array2<Float>)> {
    let mut dataset = Vec::new();

    let mut x = low;
    let mut y: Float;
    // x=16sin^3(x) y=13cos(x)-5cos(2x)-2cos(3x)-cos(4x)
    while x < high {
        y = low;

        let s: Float = x.signum() * (x / 16.0).abs().powf(1.0 / 3.0);
        let A: Float = -6.0 + 18.0 * s.powi(2) - 8.0 * s.powi(4);
        let B: Float = (11.0 + 8.0 * s.powi(2)) * (1.0 - s.powi(2)).sqrt();
        let upper = A + B;
        let lower = A - B;
        while y < high {
            if y <= upper && y >= lower {
                dataset.push((arr2(&[[x], [y]]), arr2(&[[1.0]])));
            } else {
                dataset.push((arr2(&[[x], [y]]), arr2(&[[0.0]])));
            }
            y += step;
        }
        x += step;
    }

    dataset
}

fn hue(x: Float, y: Float) -> Float {
    (y.atan2(x)).rem_euclid(TAU as Float) / TAU as Float
}

fn hsv_to_rgb(h: Float, s: Float, v: Float) -> (Float, Float, Float) {
    let c = v * s;
    let hp = h * 6.0;
    let x = c * (1.0 - ((hp % 2.0) - 1.0).abs());
    let (r1, g1, b1) = match hp as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    (r1 + m, g1 + m, b1 + m)
}

pub fn gen_rainbow_dataset(low: Float, high: Float, step: Float) -> Vec<(Array2<Float>, Array2<Float>)> {
    let mut dataset = Vec::new();

    let mut x = low;
    while x <= high {
        let mut y = low;
        while y <= high {
            let h: Float = hue(x, y);
            let (r, g, b) = hsv_to_rgb(h, 1.0, 1.0);
            dataset.push((arr2(&[[x], [y]]), arr2(&[[r], [g], [b]])));
            y += step;
        }
        x += step;
    }
    dataset
}

pub fn gen_color_dataset(low: Float, high: Float, step: Float) -> Vec<(Array2<Float>, Array2<Float>)> {
    let mut dataset = Vec::new();

    let mut x = low;
    while x <= high {
        let mut y = low;
        while y <= high {
            dataset.push((
                arr2(&[[x], [y]]),
                arr2(&[
                    [(x.powi(2)).min(255.0)],
                    [y.powi(2).min(255.0)],
                    [(x + y).powi(2).min(255.0)],
                ]),
            ));
            y += step;
        }
        x += step;
    }
    dataset
}
