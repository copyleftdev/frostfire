//! Test for the 0/1 Knapsack Problem using simulated annealing.
//!
//! The 0/1 Knapsack Problem is a classic combinatorial optimization problem:
//! Given a set of items, each with a weight and value, determine which items to include
//! in a collection so that the total weight is less than or equal to a given limit
//! and the total value is as large as possible.

use frostfire::prelude::*;
use rand::Rng;

// Seed for reproducibility as specified in the requirements
const SEED: u64 = 777;

/// An item in the knapsack problem.
#[derive(Clone, Debug)]
struct Item {
    weight: f64,
    value: f64,
}

/// A knapsack problem instance.
#[derive(Clone)]
struct KnapsackProblem {
    items: Vec<Item>,
    capacity: f64,
    known_optimal_value: Option<f64>,
}

impl KnapsackProblem {
    /// Creates a random knapsack problem.
    fn random(num_items: usize, capacity: f64, rng: &mut impl Rng) -> Self {
        let items = (0..num_items)
            .map(|_| Item {
                weight: rng.gen_range(1.0..20.0),
                value: rng.gen_range(1.0..50.0),
            })
            .collect();

        Self {
            items,
            capacity,
            known_optimal_value: None,
        }
    }

    /// Creates a knapsack problem with a known optimal solution.
    fn with_known_optimal() -> Self {
        // A small problem with a known optimal solution
        let items = vec![
            Item {
                weight: 10.0,
                value: 60.0,
            },
            Item {
                weight: 20.0,
                value: 100.0,
            },
            Item {
                weight: 30.0,
                value: 120.0,
            },
            Item {
                weight: 15.0,
                value: 80.0,
            },
            Item {
                weight: 25.0,
                value: 120.0,
            },
        ];

        // Capacity that allows specific combinations
        let capacity = 50.0;

        // Known optimal value: items 0, 1, 3 (total weight = 45, total value = 240)
        let known_optimal_value = 240.0;

        Self {
            items,
            capacity,
            known_optimal_value: Some(known_optimal_value),
        }
    }

    /// Calculates the total weight of a selection of items.
    fn total_weight(&self, selection: &[bool]) -> f64 {
        selection
            .iter()
            .zip(self.items.iter())
            .filter(|&(selected, _)| *selected)
            .map(|(_, item)| item.weight)
            .sum()
    }

    /// Calculates the total value of a selection of items.
    fn total_value(&self, selection: &[bool]) -> f64 {
        selection
            .iter()
            .zip(self.items.iter())
            .filter(|&(selected, _)| *selected)
            .map(|(_, item)| item.value)
            .sum()
    }

    /// Checks if a selection is valid (total weight <= capacity).
    fn is_valid(&self, selection: &[bool]) -> bool {
        self.total_weight(selection) <= self.capacity
    }
}

/// A state representing a selection of items for the knapsack.
#[derive(Clone)]
struct KnapsackState {
    /// Boolean vector indicating which items are selected (true) or not (false)
    selection: Vec<bool>,
}

impl KnapsackState {
    /// Creates a new random knapsack state.
    fn random(num_items: usize, rng: &mut impl Rng) -> Self {
        let selection = (0..num_items).map(|_| rng.gen_bool(0.5)).collect();

        Self { selection }
    }

    /// Creates a new empty knapsack state.
    fn empty(num_items: usize) -> Self {
        Self {
            selection: vec![false; num_items],
        }
    }
}

impl State for KnapsackState {
    fn neighbor(&self, rng: &mut impl Rng) -> Self {
        let mut new_selection = self.selection.clone();

        // Flip a random bit (include or exclude a random item)
        let idx = rng.gen_range(0..new_selection.len());
        new_selection[idx] = !new_selection[idx];

        Self {
            selection: new_selection,
        }
    }
}

/// The energy function for the knapsack problem.
///
/// Note: In this implementation, we use a penalty approach for infeasible solutions.
/// If the selection exceeds the capacity, we apply a penalty proportional to the
/// amount of constraint violation.
#[derive(Clone)]
struct KnapsackEnergy {
    problem: KnapsackProblem,
    penalty_factor: f64,
}

impl Energy for KnapsackEnergy {
    type State = KnapsackState;

    fn cost(&self, state: &Self::State) -> f64 {
        let total_value = self.problem.total_value(&state.selection);
        let total_weight = self.problem.total_weight(&state.selection);

        // We're maximizing value, but the annealer minimizes energy
        // So we negate the value to turn maximization into minimization
        let base_cost = -total_value;

        // Apply penalty if capacity is exceeded
        if total_weight > self.problem.capacity {
            // Penalty proportional to the amount of violation
            let violation = total_weight - self.problem.capacity;
            base_cost + self.penalty_factor * violation
        } else {
            base_cost
        }
    }
}

#[test]
fn test_knapsack_small_known_optimal() {
    // Create a knapsack problem with a known optimal solution
    let problem = KnapsackProblem::with_known_optimal();
    let energy = KnapsackEnergy {
        problem: problem.clone(),
        penalty_factor: 100.0, // Large penalty for constraint violations
    };

    // Start with an empty selection
    let initial_state = KnapsackState::empty(problem.items.len());

    // Set up the annealer
    let schedule = GeometricSchedule::new(50.0, 0.95);
    let mut annealer = Annealer::new(
        initial_state,
        energy,
        schedule,
        seeded_rng(SEED),
        10000, // max iterations
    );

    // Run the annealer
    let result = annealer.run_with_stats();

    // Check the best solution
    let best_selection = &result.best_state.selection;
    let best_value = problem.total_value(best_selection);
    let best_weight = problem.total_weight(best_selection);
    let is_valid = problem.is_valid(best_selection);

    println!("Best selection: {:?}", best_selection);
    println!("Best value: {}", best_value);
    println!("Best weight: {} / {}", best_weight, problem.capacity);
    println!("Is valid: {}", is_valid);

    if let Some(optimal_value) = problem.known_optimal_value {
        let ratio = best_value / optimal_value;
        println!(
            "Ratio to optimal: {:.2} ({} / {})",
            ratio, best_value, optimal_value
        );

        // The solution should achieve at least 90% of the optimal value
        assert!(
            ratio >= 0.9,
            "Solution does not achieve 90% of optimal value"
        );
    }

    // The solution must be valid (not exceed capacity)
    assert!(is_valid, "Solution exceeds capacity");
}

#[test]
fn test_knapsack_medium() {
    // Create a medium-sized knapsack problem
    let mut rng = seeded_rng(SEED);
    let num_items = 30;
    let capacity = 100.0;
    let problem = KnapsackProblem::random(num_items, capacity, &mut rng);

    let energy = KnapsackEnergy {
        problem: problem.clone(),
        penalty_factor: 100.0,
    };

    // Start with a random selection
    let initial_state = KnapsackState::random(num_items, &mut rng);
    let initial_energy = energy.cost(&initial_state);
    let initial_valid = problem.is_valid(&initial_state.selection);

    // Set up the annealer
    let schedule = GeometricSchedule::new(100.0, 0.97);
    let mut annealer = Annealer::new(
        initial_state,
        energy,
        schedule,
        seeded_rng(SEED), // New RNG with same seed
        20000,            // max iterations
    );

    // Run the annealer
    let result = annealer.run_with_stats();

    // Check the best solution
    let best_selection = &result.best_state.selection;
    let best_value = problem.total_value(best_selection);
    let best_weight = problem.total_weight(best_selection);
    let is_valid = problem.is_valid(best_selection);

    println!("Initial valid: {}", initial_valid);
    println!("Initial energy: {}", initial_energy);
    println!("Best energy: {}", result.best_energy);
    println!("Best value: {}", best_value);
    println!("Best weight: {} / {}", best_weight, problem.capacity);
    println!("Is valid: {}", is_valid);
    println!(
        "Acceptance ratio: {:.2}%",
        100.0 * result.accepted_moves as f64 / result.iterations as f64
    );

    // The solution must be valid
    assert!(is_valid, "Solution exceeds capacity");

    // The solution should be better than the initial energy
    assert!(
        result.best_energy < initial_energy,
        "Solution did not improve"
    );
}

#[test]
fn test_knapsack_large() {
    // Create a large knapsack problem
    let mut rng = seeded_rng(SEED);
    let num_items = 100;
    let capacity = 300.0;
    let problem = KnapsackProblem::random(num_items, capacity, &mut rng);

    let energy = KnapsackEnergy {
        problem: problem.clone(),
        penalty_factor: 200.0, // Higher penalty for larger problem
    };

    // Compare different schedules on the same problem

    // 1. Geometric schedule
    let initial_state = KnapsackState::empty(num_items);
    let schedule1 = GeometricSchedule::new(200.0, 0.98);
    let mut annealer1 = Annealer::new(
        initial_state.clone(),
        energy.clone(),
        schedule1,
        seeded_rng(SEED),
        30000,
    );
    let result1 = annealer1.run_with_stats();

    // 2. Logarithmic schedule
    let schedule2 = LogarithmicSchedule::new(500.0);
    let mut annealer2 = Annealer::new(
        initial_state.clone(),
        energy.clone(),
        schedule2,
        seeded_rng(SEED),
        30000,
    );
    let result2 = annealer2.run_with_stats();

    // 3. Adaptive schedule
    let schedule3 = AdaptiveSchedule::new(300.0);
    let mut annealer3 = Annealer::new(initial_state, energy, schedule3, seeded_rng(SEED), 30000);
    let result3 = annealer3.run_with_stats();

    // Compare results
    let value1 = -result1.best_energy; // Negate to get actual value
    let value2 = -result2.best_energy;
    let value3 = -result3.best_energy;

    println!("Geometric schedule value: {}", value1);
    println!("Logarithmic schedule value: {}", value2);
    println!("Adaptive schedule value: {}", value3);

    // All solutions should be valid
    assert!(
        problem.is_valid(&result1.best_state.selection),
        "Geometric solution exceeds capacity"
    );
    assert!(
        problem.is_valid(&result2.best_state.selection),
        "Logarithmic solution exceeds capacity"
    );
    assert!(
        problem.is_valid(&result3.best_state.selection),
        "Adaptive solution exceeds capacity"
    );

    // Verify that at least one schedule performs well
    let best_value = value1.max(value2).max(value3);
    println!("Best value found: {}", best_value);

    // Make sure we found a reasonable solution (not empty or trivial)
    assert!(best_value > 0.0, "No valuable solution found");
}
