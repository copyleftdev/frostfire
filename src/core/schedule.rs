//! Cooling schedules for simulated annealing.
//!
//! This module provides various cooling schedule implementations that
//! control how temperature decreases during the annealing process.

/// The `Schedule` trait defines how temperature decreases during the annealing process.
///
/// A cooling schedule is a crucial component of simulated annealing as it controls
/// the trade-off between exploration (at high temperatures) and exploitation
/// (at low temperatures) during the optimization process.
///
/// # Mathematical Background
///
/// The temperature affects the probability of accepting worse solutions:
/// - At high temperatures, the algorithm freely explores the search space
/// - As temperature decreases, the algorithm becomes more selective
/// - At very low temperatures, only improvements are accepted
///
/// # Examples
///
/// ```
/// use frostfire::prelude::*;
///
/// // Create a geometric cooling schedule
/// let schedule = GeometricSchedule::new(100.0, 0.95);
///
/// // Initial temperature
/// assert_eq!(schedule.initial_temp(), 100.0);
///
/// // Temperature decreases geometrically
/// let temp = schedule.initial_temp();
/// let next_temp = schedule.next_temp(temp, 1);
/// assert_eq!(next_temp, 100.0 * 0.95);
/// ```
pub trait Schedule: Send + Sync {
    /// Returns the initial temperature for the annealing process.
    ///
    /// # Returns
    ///
    /// The initial temperature as a positive floating-point value.
    fn initial_temp(&self) -> f64;

    /// Calculates the next temperature based on the current temperature and iteration.
    ///
    /// # Parameters
    ///
    /// * `current_temp`: The current temperature
    /// * `iteration`: The current iteration number (0-based)
    ///
    /// # Returns
    ///
    /// The next temperature as a positive floating-point value.
    fn next_temp(&self, current_temp: f64, iteration: usize) -> f64;
}

/// A geometric cooling schedule that decreases temperature by a constant factor.
///
/// This is the most commonly used cooling schedule due to its simplicity and effectiveness.
/// Temperature decreases by multiplying by a constant alpha factor (between 0 and 1)
/// at each iteration.
///
/// The temperature at iteration k is given by:
/// T(k) = T(0) * alpha^k
///
/// # Examples
///
/// ```
/// use frostfire::prelude::*;
///
/// let schedule = GeometricSchedule::new(100.0, 0.95);
/// let temp = schedule.initial_temp();
/// let next_temp = schedule.next_temp(temp, 1);
/// assert_eq!(next_temp, 100.0 * 0.95);
/// ```
#[derive(Clone, Debug)]
pub struct GeometricSchedule {
    initial_temperature: f64,
    alpha: f64,
}

impl GeometricSchedule {
    /// Creates a new geometric cooling schedule.
    ///
    /// # Parameters
    ///
    /// * `initial_temperature`: The starting temperature (must be positive)
    /// * `alpha`: The cooling rate (must be between 0 and 1, exclusive)
    ///
    /// # Panics
    ///
    /// Panics if `initial_temperature` is not positive or if `alpha` is not in (0, 1).
    pub fn new(initial_temperature: f64, alpha: f64) -> Self {
        assert!(
            initial_temperature > 0.0,
            "Initial temperature must be positive"
        );
        assert!(
            alpha > 0.0 && alpha < 1.0,
            "Alpha must be between 0 and 1 (exclusive)"
        );
        Self {
            initial_temperature,
            alpha,
        }
    }
}

impl Schedule for GeometricSchedule {
    fn initial_temp(&self) -> f64 {
        self.initial_temperature
    }

    fn next_temp(&self, current_temp: f64, _iteration: usize) -> f64 {
        current_temp * self.alpha
    }
}

/// A logarithmic cooling schedule for theoretical convergence guarantees.
///
/// This schedule decreases temperature as the inverse logarithm of the iteration,
/// ensuring theoretical convergence to the global optimum under certain conditions.
///
/// The temperature at iteration k is given by:
/// T(k) = T(0) / log(1 + k)
///
/// # Examples
///
/// ```
/// use frostfire::prelude::*;
///
/// let schedule = LogarithmicSchedule::new(100.0);
/// let temp = schedule.initial_temp();
/// let next_temp = schedule.next_temp(temp, 10);
/// assert_eq!(next_temp, 100.0 / (1.0_f64 + 10.0).ln());
/// ```
#[derive(Clone, Debug)]
pub struct LogarithmicSchedule {
    initial_temperature: f64,
}

impl LogarithmicSchedule {
    /// Creates a new logarithmic cooling schedule.
    ///
    /// # Parameters
    ///
    /// * `initial_temperature`: The starting temperature (must be positive)
    ///
    /// # Panics
    ///
    /// Panics if `initial_temperature` is not positive.
    pub fn new(initial_temperature: f64) -> Self {
        assert!(
            initial_temperature > 0.0,
            "Initial temperature must be positive"
        );
        Self {
            initial_temperature,
        }
    }
}

impl Schedule for LogarithmicSchedule {
    fn initial_temp(&self) -> f64 {
        self.initial_temperature
    }

    fn next_temp(&self, _current_temp: f64, iteration: usize) -> f64 {
        // For iteration 0, just return the initial temperature
        if iteration == 0 {
            return self.initial_temperature;
        }
        self.initial_temperature / (1.0 + iteration as f64).ln()
    }
}

/// An adaptive cooling schedule that adjusts based on observed energy changes.
///
/// This schedule dynamically adjusts the cooling rate based on the acceptance
/// ratio of proposed moves, aiming to maintain a target acceptance ratio
/// throughout the annealing process.
///
/// # Examples
///
/// ```
/// use frostfire::prelude::*;
///
/// // Create an adaptive schedule with default parameters
/// let schedule = AdaptiveSchedule::new(100.0);
///
/// // Or customize all parameters
/// let custom_schedule = AdaptiveSchedule::with_params(
///     100.0,  // initial temperature
///     0.44,   // target acceptance ratio
///     0.95,   // min alpha
///     0.99,   // max alpha
/// );
/// ```
#[derive(Clone, Debug)]
pub struct AdaptiveSchedule {
    initial_temperature: f64,
    target_acceptance_ratio: f64,
    min_alpha: f64,
    max_alpha: f64,
    acceptance_history: Vec<bool>,
    window_size: usize,
}

impl AdaptiveSchedule {
    /// Creates a new adaptive cooling schedule with default parameters.
    ///
    /// Uses a target acceptance ratio of 0.44, which is theoretically
    /// optimal for certain problem classes.
    ///
    /// # Parameters
    ///
    /// * `initial_temperature`: The starting temperature (must be positive)
    ///
    /// # Panics
    ///
    /// Panics if `initial_temperature` is not positive.
    pub fn new(initial_temperature: f64) -> Self {
        Self::with_params(initial_temperature, 0.44, 0.9, 0.99)
    }

    /// Creates a new adaptive cooling schedule with custom parameters.
    ///
    /// # Parameters
    ///
    /// * `initial_temperature`: The starting temperature (must be positive)
    /// * `target_acceptance_ratio`: The desired acceptance ratio (between 0 and 1)
    /// * `min_alpha`: The minimum cooling rate (between 0 and 1)
    /// * `max_alpha`: The maximum cooling rate (between min_alpha and 1)
    ///
    /// # Panics
    ///
    /// Panics if parameters are outside their valid ranges.
    pub fn with_params(
        initial_temperature: f64,
        target_acceptance_ratio: f64,
        min_alpha: f64,
        max_alpha: f64,
    ) -> Self {
        assert!(
            initial_temperature > 0.0,
            "Initial temperature must be positive"
        );
        assert!(
            target_acceptance_ratio > 0.0 && target_acceptance_ratio < 1.0,
            "Target acceptance ratio must be between 0 and 1"
        );
        assert!(
            min_alpha > 0.0 && min_alpha < 1.0,
            "Min alpha must be between 0 and 1"
        );
        assert!(
            max_alpha > min_alpha && max_alpha < 1.0,
            "Max alpha must be between min_alpha and 1"
        );

        Self {
            initial_temperature,
            target_acceptance_ratio,
            min_alpha,
            max_alpha,
            acceptance_history: Vec::new(),
            window_size: 100, // Use last 100 moves to compute ratio
        }
    }

    /// Records whether a proposed move was accepted.
    ///
    /// This information is used to adapt the cooling rate.
    ///
    /// # Parameters
    ///
    /// * `accepted`: Whether the move was accepted
    pub fn record_acceptance(&mut self, accepted: bool) {
        self.acceptance_history.push(accepted);
        if self.acceptance_history.len() > self.window_size {
            self.acceptance_history.remove(0);
        }
    }

    /// Computes the current acceptance ratio based on recent history.
    fn acceptance_ratio(&self) -> f64 {
        if self.acceptance_history.is_empty() {
            return 0.5; // Default if no history yet
        }
        self.acceptance_history.iter().filter(|&&x| x).count() as f64
            / self.acceptance_history.len() as f64
    }
}

impl Schedule for AdaptiveSchedule {
    fn initial_temp(&self) -> f64 {
        self.initial_temperature
    }

    fn next_temp(&self, current_temp: f64, _iteration: usize) -> f64 {
        let current_ratio = self.acceptance_ratio();

        // Adjust alpha based on how far we are from the target ratio
        let ratio_diff = current_ratio - self.target_acceptance_ratio;

        // Map the ratio difference to a cooling rate
        // - If we're accepting too many moves (ratio > target), cool faster (lower alpha)
        // - If we're accepting too few moves (ratio < target), cool slower (higher alpha)
        let alpha = if ratio_diff > 0.0 {
            // Accepting too many moves, cool faster
            self.max_alpha
                - (self.max_alpha - self.min_alpha) * (ratio_diff / self.target_acceptance_ratio)
        } else {
            // Accepting too few moves, cool slower
            self.max_alpha
        };

        // Ensure alpha stays within bounds
        let alpha = alpha.max(self.min_alpha).min(self.max_alpha);

        current_temp * alpha
    }
}
