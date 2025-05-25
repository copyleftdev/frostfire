//! Convergence analysis for simulated annealing.
//!
//! This test verifies that the annealing process exhibits proper convergence behavior,
//! specifically checking for monotonic decrease in energy over a moving average of iterations.

use frostfire::core::transition;
use frostfire::prelude::*;
use rand::Rng;

/// A simple quadratic function state for convergence testing.
///
/// This represents a point in n-dimensional space with the energy function f(x) = sum(x_i^2).
/// The global minimum is at the origin (0,0,...,0) with value 0.
#[derive(Clone)]
struct QuadraticState {
    coords: Vec<f64>,
}

impl QuadraticState {
    /// Creates a new state with random coordinates.
    fn new(dimensions: usize, range: f64, rng: &mut impl Rng) -> Self {
        let coords = (0..dimensions)
            .map(|_| rng.gen_range(-range..range))
            .collect();

        Self { coords }
    }
}

impl State for QuadraticState {
    fn neighbor(&self, rng: &mut impl Rng) -> Self {
        let mut new_coords = self.coords.clone();

        // Modify each coordinate with a small perturbation
        for coord in &mut new_coords {
            *coord += rng.gen_range(-0.1..0.1);
        }

        Self { coords: new_coords }
    }
}

/// Simple quadratic energy function.
#[derive(Clone)]
struct QuadraticEnergy;

impl Energy for QuadraticEnergy {
    type State = QuadraticState;

    fn cost(&self, state: &Self::State) -> f64 {
        state.coords.iter().map(|x| x * x).sum()
    }
}

/// Tracks energy values during annealing to analyze convergence.
#[derive(Default, Clone)]
struct EnergyTracker {
    energy_values: Vec<f64>,
}

impl EnergyTracker {
    fn new() -> Self {
        Self {
            energy_values: Vec::new(),
        }
    }

    fn add(&mut self, energy: f64) {
        self.energy_values.push(energy);
    }

    /// Calculates moving averages of energy over windows of the specified size.
    fn moving_averages(&self, window_size: usize) -> Vec<f64> {
        if self.energy_values.len() < window_size {
            return Vec::new();
        }

        let mut averages = Vec::new();
        for i in 0..=(self.energy_values.len() - window_size) {
            let sum: f64 = self.energy_values[i..(i + window_size)].iter().sum();
            averages.push(sum / window_size as f64);
        }

        averages
    }

    /// Checks if the moving averages are monotonically decreasing
    /// within the given tolerance.
    fn is_monotonic_decreasing(&self, window_size: usize, tolerance: f64) -> bool {
        let averages = self.moving_averages(window_size);
        if averages.len() < 2 {
            return true; // Not enough data points
        }

        for i in 1..averages.len() {
            // Allow small increases within tolerance
            if averages[i] > averages[i - 1] + tolerance {
                return false;
            }
        }

        true
    }

    /// Checks if energy has decreased significantly from start to end.
    fn has_significant_decrease(&self, min_decrease_ratio: f64) -> bool {
        if self.energy_values.is_empty() {
            return false;
        }

        let initial = self.energy_values[0];
        let final_val = *self.energy_values.last().unwrap();

        // Check if we've decreased by at least the specified ratio
        final_val <= initial * (1.0 - min_decrease_ratio)
    }
}

/// A custom schedule for tracking energy at each temperature change.
#[allow(dead_code)]
struct TrackingSchedule<Sch: Schedule> {
    inner: Sch,
    tracker: EnergyTracker,
    current_energy: f64,
}

impl EnergyTracker {
    fn take(&mut self) -> Self {
        std::mem::take(self)
    }
}

impl<Sch: Schedule> TrackingSchedule<Sch> {
    #[allow(dead_code)]
    fn new(schedule: Sch) -> Self {
        Self {
            inner: schedule,
            tracker: EnergyTracker::new(),
            current_energy: f64::INFINITY,
        }
    }

    #[allow(dead_code)]
    fn update_energy(&mut self, energy: f64) {
        self.current_energy = energy;
        self.tracker.add(energy);
    }

    #[allow(dead_code)]
    fn tracker(&self) -> &EnergyTracker {
        &self.tracker
    }
}

impl<Sch: Schedule> Schedule for TrackingSchedule<Sch> {
    fn initial_temp(&self) -> f64 {
        self.inner.initial_temp()
    }

    fn next_temp(&self, current_temp: f64, iteration: usize) -> f64 {
        self.inner.next_temp(current_temp, iteration)
    }
}

/// A custom annealer that tracks energy at each iteration.
struct TrackingAnnealer<S, E, Sch>
where
    S: State,
    E: Energy<State = S>,
    Sch: Schedule,
{
    inner: Annealer<S, E, Sch>,
    tracker: EnergyTracker,
}

impl<S, E, Sch> TrackingAnnealer<S, E, Sch>
where
    S: State,
    E: Energy<State = S>,
    Sch: Schedule,
{
    fn new(annealer: Annealer<S, E, Sch>) -> Self {
        Self {
            inner: annealer,
            tracker: EnergyTracker::new(),
        }
    }

    fn run(&mut self) -> (S, f64, EnergyTracker) {
        // Implementation similar to Annealer::run_with_stats but tracking energy at each step
        let initial_temp = self.inner.schedule.initial_temp();
        let mut current_temp = initial_temp;
        let mut current_state = self.inner.state.clone();
        let mut current_energy = self.inner.energy.cost(&current_state);

        // Track initial energy
        self.tracker.add(current_energy);

        // Save the initial state as the best state
        let mut best_state = current_state.clone();
        let mut best_energy = current_energy;

        // Main annealing loop
        for i in 0..self.inner.max_iters {
            // Generate a neighboring state
            let new_state = current_state.neighbor(&mut self.inner.rng);
            let new_energy = self.inner.energy.cost(&new_state);

            // Calculate the energy difference
            let delta = new_energy - current_energy;

            // Decide whether to accept the new state
            if transition::accept(delta, current_temp, &mut self.inner.rng) {
                // Accept the new state
                current_state = new_state;
                current_energy = new_energy;

                // Track energy at each accepted step
                self.tracker.add(current_energy);

                // Update the best state if we found a better one
                if new_energy < best_energy {
                    best_state = current_state.clone();
                    best_energy = new_energy;
                }
            }

            // Update the temperature according to the cooling schedule
            current_temp = self.inner.schedule.next_temp(current_temp, i);
        }

        (best_state, best_energy, self.tracker.take())
    }
}

#[test]
fn test_convergence_monotonic() {
    // Test that energy decreases monotonically over time
    let dimensions = 10;
    let mut rng = seeded_rng(42);
    let initial_state = QuadraticState::new(dimensions, 10.0, &mut rng);
    let energy = QuadraticEnergy;

    // Create a tracking annealer
    let schedule = GeometricSchedule::new(10.0, 0.95);
    let annealer = Annealer::new(initial_state, energy, schedule, rng, 5000);

    let mut tracking_annealer = TrackingAnnealer::new(annealer);
    let (best_state, best_energy, tracker) = tracking_annealer.run();

    println!("Best energy: {}", best_energy);
    println!("Number of energy samples: {}", tracker.energy_values.len());

    // Check if energy has decreased significantly
    assert!(
        tracker.has_significant_decrease(0.9),
        "Energy did not decrease significantly"
    );

    // Check if moving averages are monotonically decreasing (with some tolerance)
    assert!(
        tracker.is_monotonic_decreasing(50, 0.1),
        "Energy is not monotonically decreasing over time"
    );

    // The best solution should be close to the origin
    for &x in &best_state.coords {
        assert!(
            x.abs() < 1.0,
            "Solution coordinate {} is not close to origin",
            x
        );
    }
}

#[test]
fn test_convergence_different_schedules() {
    // Compare convergence behavior of different cooling schedules
    let dimensions = 5;
    let mut rng = seeded_rng(42);
    let initial_state = QuadraticState::new(dimensions, 10.0, &mut rng);
    let energy = QuadraticEnergy;

    // Run with geometric schedule
    let geo_schedule = GeometricSchedule::new(10.0, 0.95);
    let geo_annealer = Annealer::new(
        initial_state.clone(),
        energy.clone(),
        geo_schedule,
        seeded_rng(42), // Same seed for fair comparison
        3000,
    );
    let mut tracking_geo = TrackingAnnealer::new(geo_annealer);
    let (_, geo_energy, geo_tracker) = tracking_geo.run();

    // Run with logarithmic schedule
    let log_schedule = LogarithmicSchedule::new(10.0);
    let log_annealer = Annealer::new(
        initial_state.clone(),
        energy.clone(),
        log_schedule,
        seeded_rng(42),
        3000,
    );
    let mut tracking_log = TrackingAnnealer::new(log_annealer);
    let (_, log_energy, log_tracker) = tracking_log.run();

    // Run with adaptive schedule
    let adp_schedule = AdaptiveSchedule::new(10.0);
    let adp_annealer = Annealer::new(initial_state, energy, adp_schedule, seeded_rng(42), 3000);
    let mut tracking_adp = TrackingAnnealer::new(adp_annealer);
    let (_, adp_energy, adp_tracker) = tracking_adp.run();

    println!("Geometric schedule final energy: {}", geo_energy);
    println!("Logarithmic schedule final energy: {}", log_energy);
    println!("Adaptive schedule final energy: {}", adp_energy);

    // All schedules should show significant energy decrease
    assert!(
        geo_tracker.has_significant_decrease(0.8),
        "Geometric schedule did not show significant energy decrease"
    );
    assert!(
        log_tracker.has_significant_decrease(0.8),
        "Logarithmic schedule did not show significant energy decrease"
    );
    assert!(
        adp_tracker.has_significant_decrease(0.8),
        "Adaptive schedule did not show significant energy decrease"
    );

    // Compare convergence rates - which schedule converged fastest
    let geo_final_avg = geo_tracker
        .moving_averages(20)
        .last()
        .cloned()
        .unwrap_or(f64::INFINITY);
    let log_final_avg = log_tracker
        .moving_averages(20)
        .last()
        .cloned()
        .unwrap_or(f64::INFINITY);
    let adp_final_avg = adp_tracker
        .moving_averages(20)
        .last()
        .cloned()
        .unwrap_or(f64::INFINITY);

    println!("Geometric schedule final average: {}", geo_final_avg);
    println!("Logarithmic schedule final average: {}", log_final_avg);
    println!("Adaptive schedule final average: {}", adp_final_avg);

    // At least one schedule should achieve a good solution
    let best_energy = geo_energy.min(log_energy).min(adp_energy);
    assert!(best_energy < 0.1, "No schedule achieved a good solution");
}

#[test]
fn test_convergence_restart() {
    // Test convergence with multiple restarts from different initial points
    let dimensions = 3;
    let num_restarts = 5;
    let energy = QuadraticEnergy;

    let mut best_overall_energy = f64::INFINITY;
    let mut all_energies = Vec::new();

    for i in 0..num_restarts {
        // Start from a different random point each time
        let restart_seed = 42 + i as u64;
        let mut restart_rng = seeded_rng(restart_seed);
        let initial_state = QuadraticState::new(dimensions, 10.0, &mut restart_rng);

        let schedule = GeometricSchedule::new(5.0, 0.97);
        let mut annealer = Annealer::new(
            initial_state,
            energy.clone(),
            schedule,
            restart_rng,
            1000, // Shorter runs for each restart
        );

        let (_, energy_value) = annealer.run();
        all_energies.push(energy_value);

        if energy_value < best_overall_energy {
            best_overall_energy = energy_value;
        }

        println!("Restart {}: final energy = {}", i, energy_value);
    }

    println!("Best energy across all restarts: {}", best_overall_energy);

    // Check that at least one restart found a good solution
    assert!(
        best_overall_energy < 0.1,
        "Multiple restarts failed to find a good solution"
    );

    // Check that we get different results from different starting points
    let mut has_different_results = false;
    for i in 1..all_energies.len() {
        if (all_energies[i] - all_energies[0]).abs() > 1e-10 {
            has_different_results = true;
            break;
        }
    }

    assert!(
        has_different_results,
        "Different starting points led to identical results"
    );
}
