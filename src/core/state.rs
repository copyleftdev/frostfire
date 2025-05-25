//! State representation for simulated annealing.
//!
//! The `State` trait represents a candidate solution in the search space.
//! It provides methods for generating neighboring states during the annealing process.

use rand::Rng;

/// The `State` trait defines the representation of a candidate solution
/// in the simulated annealing process.
///
/// Implementors must provide a method to generate a neighboring state,
/// which is a slight modification of the current state according to some
/// problem-specific rule.
///
/// # Examples
///
/// ```
/// use frostfire::prelude::*;
/// use rand::Rng;
///
/// #[derive(Clone)]
/// struct VectorState(Vec<f64>);
///
/// impl State for VectorState {
///     fn neighbor(&self, rng: &mut impl Rng) -> Self {
///         let mut new_state = self.clone();
///         let idx = rng.gen_range(0..new_state.0.len());
///         new_state.0[idx] += rng.gen_range(-0.1..0.1);
///         new_state
///     }
/// }
/// ```
pub trait State: Clone + Send + Sync {
    /// Generate a neighboring state by making a small modification to the current state.
    ///
    /// The neighboring state should be a small perturbation of the current state,
    /// allowing the annealing process to explore the local search space effectively.
    ///
    /// # Parameters
    ///
    /// * `rng`: A random number generator used to introduce randomness in the neighbor generation.
    ///
    /// # Returns
    ///
    /// A new state that is a neighbor of the current state.
    fn neighbor(&self, rng: &mut impl Rng) -> Self;
}
