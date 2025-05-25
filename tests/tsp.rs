//! Test for the Traveling Salesman Problem (TSP) using simulated annealing.
//!
//! This test verifies that the frostfire library can effectively solve
//! the TSP problem, ensuring convergence to within 1.2× of known
//! optimal solutions on symmetric distance matrices.

use frostfire::prelude::*;
use rand::Rng;
use std::fmt;

// Seed for reproducibility
const SEED: u64 = 42;

/// Represents a TSP problem as a vector of cities with (x, y) coordinates.
#[derive(Clone)]
struct TspProblem {
    cities: Vec<(f64, f64)>,
    #[allow(dead_code)]
    optimal_tour: Option<Vec<usize>>,
    optimal_distance: Option<f64>,
}

impl TspProblem {
    /// Creates a new TSP problem with random city positions.
    fn random(num_cities: usize, rng: &mut impl Rng) -> Self {
        let cities = (0..num_cities)
            .map(|_| (rng.gen_range(0.0..100.0), rng.gen_range(0.0..100.0)))
            .collect();

        Self {
            cities,
            optimal_tour: None,
            optimal_distance: None,
        }
    }

    /// Creates a simple TSP problem with a known optimal solution.
    fn with_known_optimal() -> Self {
        // A small problem with a known optimal solution
        let cities = vec![
            (0.0, 0.0), // 0
            (0.0, 5.0), // 1
            (5.0, 5.0), // 2
            (5.0, 0.0), // 3
        ];

        // Optimal tour: 0-1-2-3-0
        let optimal_tour = vec![0, 1, 2, 3];

        // Optimal distance: 5 + 5 + 5 + 5 = 20
        let optimal_distance = 20.0;

        Self {
            cities,
            optimal_tour: Some(optimal_tour),
            optimal_distance: Some(optimal_distance),
        }
    }

    /// Calculates the distance between two cities.
    fn distance(&self, city1: usize, city2: usize) -> f64 {
        let (x1, y1) = self.cities[city1];
        let (x2, y2) = self.cities[city2];

        ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt()
    }

    /// Calculates the total distance of a tour.
    fn tour_distance(&self, tour: &[usize]) -> f64 {
        let mut total = 0.0;

        for i in 0..tour.len() {
            let from = tour[i];
            let to = tour[(i + 1) % tour.len()];
            total += self.distance(from, to);
        }

        total
    }
}

/// A state representing a tour in the TSP problem.
#[derive(Clone)]
struct TspState {
    tour: Vec<usize>,
}

impl fmt::Debug for TspState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tour: {:?}", self.tour)
    }
}

impl TspState {
    /// Creates a new TSP state with a random tour.
    fn random(num_cities: usize, rng: &mut impl Rng) -> Self {
        let mut tour: Vec<usize> = (0..num_cities).collect();
        // Fisher-Yates shuffle for random permutation
        for i in (1..num_cities).rev() {
            let j = rng.gen_range(0..=i);
            tour.swap(i, j);
        }

        Self { tour }
    }
}

impl State for TspState {
    fn neighbor(&self, rng: &mut impl Rng) -> Self {
        // Create a neighbor by swapping two random cities
        let mut new_tour = self.tour.clone();
        let idx1 = rng.gen_range(0..new_tour.len());
        let idx2 = rng.gen_range(0..new_tour.len());

        if idx1 != idx2 {
            new_tour.swap(idx1, idx2);
        }

        Self { tour: new_tour }
    }
}

/// The energy function for the TSP problem.
struct TspEnergy {
    problem: TspProblem,
}

impl Energy for TspEnergy {
    type State = TspState;

    fn cost(&self, state: &Self::State) -> f64 {
        self.problem.tour_distance(&state.tour)
    }
}

#[test]
fn test_tsp_small_known_optimal() {
    // Create a TSP problem with a known optimal solution
    let problem = TspProblem::with_known_optimal();
    let energy = TspEnergy {
        problem: problem.clone(),
    };

    // Initial state - random tour
    let mut rng = seeded_rng(SEED);
    let initial_state = TspState::random(problem.cities.len(), &mut rng);

    // Set up the annealer
    let schedule = GeometricSchedule::new(100.0, 0.95);
    let mut annealer = Annealer::new(
        initial_state,
        energy,
        schedule,
        rng,
        10000, // max iterations
    );

    // Run the annealer
    let (best_state, best_energy) = annealer.run();

    // Check that we found the optimal solution or close to it
    let optimal_distance = problem.optimal_distance.unwrap();
    let ratio = best_energy / optimal_distance;

    println!("Best tour: {:?}", best_state.tour);
    println!("Best distance: {}", best_energy);
    println!("Optimal distance: {}", optimal_distance);
    println!("Ratio: {}", ratio);

    // The solution should be within 1.2× of the optimal
    assert!(ratio <= 1.2, "Solution is not within 1.2× of optimal");
}

#[test]
fn test_tsp_medium() {
    // Create a medium-sized TSP problem
    let mut rng = seeded_rng(SEED);
    let num_cities = 20;
    let problem = TspProblem::random(num_cities, &mut rng);
    let energy = TspEnergy {
        problem: problem.clone(),
    };

    // Initial state - random tour
    let initial_state = TspState::random(num_cities, &mut rng);
    let initial_energy = energy.cost(&initial_state);

    // Set up the annealer
    let schedule = GeometricSchedule::new(100.0, 0.95);
    let mut annealer = Annealer::new(
        initial_state,
        energy,
        schedule,
        seeded_rng(SEED), // New RNG with same seed
        20000,            // max iterations
    );

    // Run the annealer
    let result = annealer.run_with_stats();

    println!("Initial energy: {}", initial_energy);
    println!("Final energy: {}", result.final_energy);
    println!("Best energy: {}", result.best_energy);
    println!(
        "Improvement: {:.2}%",
        100.0 * (initial_energy - result.best_energy) / initial_energy
    );
    println!(
        "Acceptance ratio: {:.2}%",
        100.0 * result.accepted_moves as f64 / result.iterations as f64
    );

    // The solution should be significantly better than the initial random tour
    assert!(
        result.best_energy < 0.5 * initial_energy,
        "Solution did not improve significantly"
    );
}

#[test]
fn test_tsp_large() {
    // Create a large TSP problem
    let mut rng = seeded_rng(SEED);
    let num_cities = 50;
    let problem = TspProblem::random(num_cities, &mut rng);
    let energy = TspEnergy {
        problem: problem.clone(),
    };

    // Initial state - random tour
    let initial_state = TspState::random(num_cities, &mut rng);
    let initial_energy = energy.cost(&initial_state);

    // Set up the annealer
    let schedule = GeometricSchedule::new(1000.0, 0.98);
    let mut annealer = Annealer::new(
        initial_state,
        energy,
        schedule,
        seeded_rng(SEED), // New RNG with same seed
        50000,            // max iterations for larger problem
    );

    // Run the annealer
    let result = annealer.run_with_stats();

    println!("Initial energy: {}", initial_energy);
    println!("Final energy: {}", result.final_energy);
    println!("Best energy: {}", result.best_energy);
    println!(
        "Improvement: {:.2}%",
        100.0 * (initial_energy - result.best_energy) / initial_energy
    );
    println!(
        "Acceptance ratio: {:.2}%",
        100.0 * result.accepted_moves as f64 / result.iterations as f64
    );

    // The solution should be significantly better than the initial random tour
    assert!(
        result.best_energy < 0.4 * initial_energy,
        "Solution did not improve significantly"
    );
}
