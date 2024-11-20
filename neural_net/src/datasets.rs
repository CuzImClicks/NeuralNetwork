use ndarray::{arr2, Array2};

pub fn gen_x2_dataset(low: f64, high: f64, step: f64) -> Vec<(Array2<f64>, Array2<f64>)> {

    let mut v: Vec<(Array2<f64>, Array2<f64>)> = Vec::with_capacity((high / step) as usize);
    let mut i = low;
    while i <= high {
        v.push((arr2(&[[i]]), arr2(&[[i.powi(2)]])));
        i += step;
    }
    v
}

pub fn gen_xor_dataset() -> Vec<(Array2<f64>, Array2<f64>)> {
    vec![
        (arr2(&[[0.0], [0.0]]), arr2(&[[0.0]])),
        (arr2(&[[0.0], [1.0]]), arr2(&[[1.0]])),
        (arr2(&[[1.0], [0.0]]), arr2(&[[1.0]])),
        (arr2(&[[1.0], [1.0]]), arr2(&[[0.0]])),
    ]
}

pub fn gen_circle_dataset(low: f64, high: f64, step: f64) -> Vec<(Array2<f64>, Array2<f64>)> {
    let mut dataset = Vec::new();

    let mut x = low;
    while x <= high {
        let mut y = low;
        while y <= high {
            let distance = ((x - 2.0).powi(2) + (y - 2.0).powi(2)).sqrt();

            let label = if distance < 1.0 { 1.0 } else { 0.0 };

            dataset.push((
                arr2(&[[x], [y]]), 
                arr2(&[[label]]),  
            ));

            y += step;
        }
        x += step;
    }

    dataset
}
