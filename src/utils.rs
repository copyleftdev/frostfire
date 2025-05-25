//! Utility functions for the frostfire library.
//!
//! This module provides various helper functions and utilities
//! that may be useful when working with simulated annealing.

use std::time::{Duration, Instant};

/// Measures the execution time of a function.
///
/// This is useful for benchmarking different annealing configurations.
///
/// # Parameters
///
/// * `f`: A function to execute and measure
///
/// # Returns
///
/// A tuple containing the function's return value and the execution duration.
///
/// # Examples
///
/// ```
/// use frostfire::utils::time_execution;
/// use std::thread::sleep;
/// use std::time::Duration;
///
/// let (result, duration) = time_execution(|| {
///     sleep(Duration::from_millis(10));
///     42
/// });
///
/// assert_eq!(result, 42);
/// assert!(duration.as_millis() >= 10);
/// ```
pub fn time_execution<F, T>(f: F) -> (T, Duration)
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();
    (result, duration)
}

/// Calculates the average of a slice of f64 values.
///
/// # Parameters
///
/// * `values`: A slice of f64 values
///
/// # Returns
///
/// The average of the values, or 0.0 if the slice is empty.
///
/// # Examples
///
/// ```
/// use frostfire::utils::average;
///
/// let values = [1.0, 2.0, 3.0, 4.0, 5.0];
/// assert_eq!(average(&values), 3.0);
/// ```
pub fn average(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Calculates the standard deviation of a slice of f64 values.
///
/// # Parameters
///
/// * `values`: A slice of f64 values
///
/// # Returns
///
/// The standard deviation of the values, or 0.0 if the slice has fewer than 2 elements.
///
/// # Examples
///
/// ```
/// use frostfire::utils::standard_deviation;
///
/// let values = [1.0, 2.0, 3.0, 4.0, 5.0];
/// assert!((standard_deviation(&values) - 1.581139).abs() < 1e-6);
/// ```
pub fn standard_deviation(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let avg = average(values);
    let variance =
        values.iter().map(|&x| (x - avg).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

    variance.sqrt()
}

/// Computes the Boltzmann acceptance probability.
///
/// This is the probability used in the Metropolis acceptance criterion.
///
/// # Parameters
///
/// * `delta`: The energy difference (new_energy - current_energy)
/// * `temperature`: The current temperature
///
/// # Returns
///
/// The acceptance probability as a value between 0 and 1.
///
/// # Examples
///
/// ```
/// use frostfire::utils::boltzmann_probability;
///
/// // Improvements always have probability 1
/// assert_eq!(boltzmann_probability(-1.0, 1.0), 1.0);
///
/// // Worse solutions have lower probability
/// let p = boltzmann_probability(2.0, 1.0);
/// assert!(p > 0.0 && p < 1.0);
///
/// // Higher temperature increases acceptance probability
/// assert!(boltzmann_probability(1.0, 2.0) > boltzmann_probability(1.0, 1.0));
/// ```
pub fn boltzmann_probability(delta: f64, temperature: f64) -> f64 {
    if delta <= 0.0 {
        1.0
    } else {
        (-delta / temperature).exp()
    }
}
