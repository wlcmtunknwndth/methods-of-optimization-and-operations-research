use nalgebra::DVector;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub type ObjectiveFn = dyn Fn(&DVector<f64>) -> f64;
pub type GradientFn = dyn Fn(&DVector<f64>) -> DVector<f64>;

#[derive(Debug)]
pub struct OptimizerResult {
    pub x: DVector<f64>,
    pub f_x: f64,
    pub iterations: usize,
    pub history: Vec<(f64, f64, f64)>,
    pub terminated_early: bool,
}

pub fn gradient_descent(
    initial_point: DVector<f64>,
    f: &ObjectiveFn,
    grad: &GradientFn,
    initial_step: f64,
    step_decay: f64,
    step_increase: f64,
    tolerance: f64,
    max_iterations: usize,
    stop_flag: Arc<AtomicBool>,
) -> OptimizerResult {
    let mut x = initial_point;
    let mut f_x = f(&x);
    let mut iter = 0;
    let mut step = initial_step;

    let mut history = Vec::new();
    history.push((x[0], x[1], f_x));

    while iter < max_iterations {
        if stop_flag.load(Ordering::SeqCst) {
            return OptimizerResult {
                x,
                f_x,
                iterations: iter,
                history,
                terminated_early: true,
            };
        }

        let g = grad(&x);

        if g.norm() < tolerance {
            break;
        }

        let direction = -g;

        // Адаптивный выбор шага
        let mut found_step = false;
        let mut trial_step = step;

        for _ in 0..20 {
            let x_trial = &x + trial_step * &direction;
            let f_trial = f(&x_trial);

            if f_trial < f_x {
                x = x_trial;
                f_x = f_trial;
                step = (step_increase * trial_step).min(1.0);
                found_step = true;
                break;
            } else {
                trial_step *= step_decay;
            }
        }

        if !found_step {
            break;
        }

        iter += 1;
        history.push((x[0], x[1], f_x));
    }

    OptimizerResult {
        x,
        f_x,
        iterations: iter,
        history,
        terminated_early: false,
    }
}
