//! Energy (cost function) representation for simulated annealing.
//!
//! The `Energy` trait defines the cost function to be minimized
//! during the simulated annealing process.

use crate::core::state::State;

/// The `Energy` trait defines the cost function to be minimized
/// during the simulated annealing process.
///
/// In simulated annealing, we seek to find a state that minimizes
/// the energy (cost) function. This trait associates a numerical
/// value with each state, allowing the annealer to compare different
/// states and guide the optimization process.
///
/// # Examples
///
/// ```
/// use frostfire::prelude::*;
///
/// #[derive(Clone)]
/// struct VectorState(Vec<f64>);
///
/// impl State for VectorState {
///     // Implementation omitted for brevity
///     # fn neighbor(&self, rng: &mut impl rand::Rng) -> Self { self.clone() }
/// }
///
/// struct QuadraticEnergy;
///
/// impl Energy for QuadraticEnergy {
///     type State = VectorState;
///
///     fn cost(&self, state: &Self::State) -> f64 {
///         // Simple quadratic function: sum of squares
///         state.0.iter().map(|x| x * x).sum()
///     }
/// }
/// ```
pub trait Energy {
    /// The type of state this energy function evaluates.
    type State: State;

    /// Calculates the cost (energy) of a given state.
    ///
    /// This function associates a numerical value with each state,
    /// which the annealer seeks to minimize.
    ///
    /// # Parameters
    ///
    /// * `state`: The state to evaluate
    ///
    /// # Returns
    ///
    /// The cost (energy) of the given state as a floating-point value.
    /// Lower values are considered better in the annealing process.
    fn cost(&self, state: &Self::State) -> f64;
}
