//! # Frostfire
//!
//! A modular, mathematically rigorous, performant, reusable simulated annealing optimization engine.
//!
//! ## Overview
//!
//! Simulated annealing is a probabilistic technique for approximating the global optimum
//! of a given function. It is often used when the search space is discrete and finding
//! an approximate global optimum is more important than finding a precise local optimum.
//!
//! This library provides a modular framework for implementing simulated annealing solutions
//! with a focus on:
//!
//! - Mathematical rigor and correctness
//! - Deterministic behavior (when seeded)
//! - Performance through zero-cost abstractions
//! - Modular and reusable components
//!
//! ## Core Components
//!
//! - `State`: Represents a candidate solution in the search space
//! - `Energy`: Defines the cost function to be minimized
//! - `Schedule`: Controls the cooling process during annealing
//! - `Annealer`: The main engine that performs the optimization
//!
//! ## Example
//!
//! ```rust
//! use frostfire::prelude::*;
//! use rand::Rng;
//!
//! // Define your problem state
//! #[derive(Clone)]
//! struct MyState(Vec<f64>);
//!
//! impl State for MyState {
//!     fn neighbor(&self, rng: &mut impl Rng) -> Self {
//!         let mut new_state = self.clone();
//!         let idx = rng.gen_range(0..new_state.0.len());
//!         new_state.0[idx] += rng.gen_range(-0.1..0.1);
//!         new_state
//!     }
//! }
//!
//! // Define your energy/cost function
//! struct MyEnergy;
//!
//! impl Energy for MyEnergy {
//!     type State = MyState;
//!
//!     fn cost(&self, state: &Self::State) -> f64 {
//!         // Simple quadratic function
//!         state.0.iter().map(|x| x * x).sum()
//!     }
//! }
//!
//! // Run the annealer (not executed in doc tests)
//! # fn main() {
//! let initial_state = MyState(vec![1.0, 1.0, 1.0]);
//! let energy = MyEnergy;
//! let schedule = GeometricSchedule::new(100.0, 0.95);
//!
//! let mut annealer = Annealer::new(
//!     initial_state,
//!     energy,
//!     schedule,
//!     seeded_rng(42),
//!     10000,
//! );
//!
//! let (best_state, best_energy) = annealer.run();
//! # }
//! ```

pub mod core;
pub mod prelude;
pub mod rng;
pub mod utils;

// Re-export core components for convenient access
pub use crate::core::annealer::Annealer;
pub use crate::core::energy::Energy;
pub use crate::core::schedule::{
    AdaptiveSchedule, GeometricSchedule, LogarithmicSchedule, Schedule,
};
pub use crate::core::state::State;
pub use crate::core::transition;
pub use crate::rng::seeded_rng::seeded_rng;
