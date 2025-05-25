//! Transition policies for simulated annealing.
//!
//! This module provides functions to determine whether a proposed state
//! transition should be accepted during the annealing process.

use rand::Rng;

/// The classic Metropolis-Hastings acceptance criterion for simulated annealing.
///
/// This function implements the standard acceptance probability function:
/// - If the new state has lower energy (delta < 0), accept it with probability 1
/// - If the new state has higher energy (delta >= 0), accept it with probability exp(-delta/temperature)
///
/// This allows the algorithm to occasionally accept worse solutions, helping it
/// escape local minima. As the temperature decreases, the probability of accepting
/// worse solutions also decreases, allowing the algorithm to converge.
///
/// # Mathematical Background
///
/// The acceptance probability is given by:
///
/// P(accept) = min(1, exp(-delta/T))
///
/// where:
/// - delta is the energy difference (new_energy - current_energy)
/// - T is the current temperature
///
/// # Parameters
///
/// * `delta`: The energy difference (new_energy - current_energy)
/// * `temperature`: The current temperature in the annealing process
/// * `rng`: A random number generator
///
/// # Returns
///
/// `true` if the transition should be accepted, `false` otherwise.
///
/// # Examples
///
/// ```
/// use frostfire::core::transition::accept;
/// use rand::thread_rng;
///
/// let mut rng = thread_rng();
///
/// // Always accept improvements
/// assert!(accept(-10.0, 1.0, &mut rng));
///
/// // May sometimes accept worse solutions, especially at high temperatures
/// let accept_count = (0..1000)
///     .filter(|_| accept(5.0, 10.0, &mut rng))
///     .count();
///
/// // At T=10.0 and delta=5.0, acceptance probability should be around 60%
/// assert!(accept_count > 500 && accept_count < 700);
/// ```
pub fn accept(delta: f64, temperature: f64, rng: &mut impl Rng) -> bool {
    if delta < 0.0 {
        true
    } else {
        rng.gen::<f64>() < (-delta / temperature).exp()
    }
}
