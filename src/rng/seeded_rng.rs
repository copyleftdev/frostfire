//! Deterministic random number generation for reproducible simulated annealing.
//!
//! This module provides functionality for creating seeded random number generators
//! to ensure that simulation runs are reproducible.

use rand::rngs::StdRng;
use rand::SeedableRng;

/// Creates a seeded random number generator for deterministic simulations.
///
/// Using a seeded RNG is crucial for reproducibility in simulated annealing.
/// By providing the same seed, you can ensure that the same sequence of random
/// numbers is generated, making the annealing process deterministic and reproducible.
///
/// # Parameters
///
/// * `seed`: The seed value to initialize the random number generator
///
/// # Returns
///
/// A seeded `StdRng` instance that can be used for deterministic random number generation.
///
/// # Examples
///
/// ```
/// use frostfire::rng::seeded_rng::seeded_rng;
///
/// // Create a deterministic RNG with seed 42
/// let rng = seeded_rng(42);
///
/// // Using the same seed will always produce the same sequence of random numbers
/// let rng1 = seeded_rng(123);
/// let rng2 = seeded_rng(123);
/// // rng1 and rng2 will generate identical sequences
/// ```
pub fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}
