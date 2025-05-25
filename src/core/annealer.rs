//! Main annealing engine implementation.
//!
//! This module provides the core annealing algorithm that drives the optimization process.

use crate::core::energy::Energy;
use crate::core::schedule::Schedule;
use crate::core::state::State;
use crate::core::transition;
use rand::rngs::StdRng;
use std::fmt;

/// Results from an annealing run, containing detailed statistics and the best solution found.
#[derive(Clone)]
pub struct AnnealingResult<S: State> {
    /// The best state found during the annealing process
    pub best_state: S,
    /// The energy (cost) of the best state
    pub best_energy: f64,
    /// The final state after annealing (may not be the best state)
    pub final_state: S,
    /// The energy (cost) of the final state
    pub final_energy: f64,
    /// The number of iterations performed
    pub iterations: usize,
    /// The number of accepted moves
    pub accepted_moves: usize,
    /// The number of rejected moves
    pub rejected_moves: usize,
    /// The initial temperature
    pub initial_temp: f64,
    /// The final temperature
    pub final_temp: f64,
}

impl<S: State> fmt::Debug for AnnealingResult<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnnealingResult")
            .field("best_energy", &self.best_energy)
            .field("final_energy", &self.final_energy)
            .field("iterations", &self.iterations)
            .field("accepted_moves", &self.accepted_moves)
            .field("rejected_moves", &self.rejected_moves)
            .field("acceptance_ratio", &(self.accepted_moves as f64 / self.iterations as f64))
            .field("initial_temp", &self.initial_temp)
            .field("final_temp", &self.final_temp)
            .finish()
    }
}

/// Main annealer engine that performs simulated annealing optimization.
///
/// The `Annealer` encapsulates all components needed for simulated annealing:
/// - A state representation
/// - An energy function to be minimized
/// - A cooling schedule
/// - A random number generator
/// - Termination criteria
///
/// # Examples
///
/// ```
/// use frostfire::prelude::*;
/// use rand::Rng;
///
/// // Define a simple optimization problem
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
///
/// struct QuadraticEnergy;
///
/// impl Energy for QuadraticEnergy {
///     type State = VectorState;
///
///     fn cost(&self, state: &Self::State) -> f64 {
///         state.0.iter().map(|x| x * x).sum()
///     }
/// }
///
/// // Create and run the annealer
/// let initial_state = VectorState(vec![1.0, 1.0, 1.0]);
/// let energy = QuadraticEnergy;
/// let schedule = GeometricSchedule::new(100.0, 0.95);
///
/// let mut annealer = Annealer::new(
///     initial_state,
///     energy,
///     schedule,
///     seeded_rng(42),
///     10000,
/// );
///
/// let (best_state, best_energy) = annealer.run();
/// ```
pub struct Annealer<S, E, Sch>
where
    S: State,
    E: Energy<State = S>,
    Sch: Schedule,
{
    /// The current state in the annealing process
    pub state: S,
    /// The energy function to be minimized
    pub energy: E,
    /// The cooling schedule
    pub schedule: Sch,
    /// The random number generator
    pub rng: StdRng,
    /// The maximum number of iterations
    pub max_iters: usize,
    /// The best state found so far
    best_state: Option<S>,
    /// The energy of the best state
    best_energy: f64,
    /// Whether to track detailed statistics
    collect_stats: bool,
    /// Number of accepted moves
    accepted_moves: usize,
    /// Number of rejected moves
    rejected_moves: usize,
}

impl<S, E, Sch> Annealer<S, E, Sch>
where
    S: State,
    E: Energy<State = S>,
    Sch: Schedule,
{
    /// Creates a new annealer with the given components.
    ///
    /// # Parameters
    ///
    /// * `initial_state`: The starting state for the annealing process
    /// * `energy`: The energy function to be minimized
    /// * `schedule`: The cooling schedule
    /// * `rng`: A seeded random number generator for reproducibility
    /// * `max_iters`: The maximum number of iterations
    ///
    /// # Examples
    ///
    /// ```
    /// use frostfire::prelude::*;
    ///
    /// // Implementation details omitted for brevity
    /// # #[derive(Clone)]
    /// # struct MyState;
    /// # impl State for MyState {
    /// #     fn neighbor(&self, _: &mut impl rand::Rng) -> Self { self.clone() }
    /// # }
    /// # struct MyEnergy;
    /// # impl Energy for MyEnergy {
    /// #     type State = MyState;
    /// #     fn cost(&self, _: &Self::State) -> f64 { 0.0 }
    /// # }
    ///
    /// let annealer = Annealer::new(
    ///     MyState,
    ///     MyEnergy,
    ///     GeometricSchedule::new(100.0, 0.95),
    ///     seeded_rng(42),
    ///     10000,
    /// );
    /// ```
    pub fn new(initial_state: S, energy: E, schedule: Sch, rng: StdRng, max_iters: usize) -> Self {
        let initial_energy = energy.cost(&initial_state);
        Self {
            state: initial_state,
            energy,
            schedule,
            rng,
            max_iters,
            best_state: None,
            best_energy: initial_energy,
            collect_stats: false,
            accepted_moves: 0,
            rejected_moves: 0,
        }
    }

    /// Enables collection of detailed statistics during the annealing process.
    ///
    /// When enabled, the annealer will track additional information such as
    /// the number of accepted and rejected moves, which can be helpful for
    /// analyzing and tuning the annealing process.
    ///
    /// # Returns
    ///
    /// The modified annealer with statistics collection enabled.
    pub fn with_stats(mut self) -> Self {
        self.collect_stats = true;
        self
    }

    /// Runs the annealing process to completion.
    ///
    /// This method performs the simulated annealing algorithm until the
    /// maximum number of iterations is reached.
    ///
    /// # Returns
    ///
    /// A tuple containing the best state found and its energy.
    ///
    /// # Examples
    ///
    /// ```
    /// use frostfire::prelude::*;
    ///
    /// // Implementation details omitted for brevity
    /// # #[derive(Clone)]
    /// # struct MyState;
    /// # impl State for MyState {
    /// #     fn neighbor(&self, _: &mut impl rand::Rng) -> Self { self.clone() }
    /// # }
    /// # struct MyEnergy;
    /// # impl Energy for MyEnergy {
    /// #     type State = MyState;
    /// #     fn cost(&self, _: &Self::State) -> f64 { 0.0 }
    /// # }
    ///
    /// let mut annealer = Annealer::new(
    ///     MyState,
    ///     MyEnergy,
    ///     GeometricSchedule::new(100.0, 0.95),
    ///     seeded_rng(42),
    ///     10000,
    /// );
    ///
    /// let (best_state, best_energy) = annealer.run();
    /// ```
    pub fn run(&mut self) -> (S, f64) {
        let result = self.run_with_stats();
        (result.best_state, result.best_energy)
    }

    /// Runs the annealing process and returns detailed statistics.
    ///
    /// This method is similar to `run()` but returns a more detailed result
    /// structure containing statistics about the annealing process.
    ///
    /// # Returns
    ///
    /// An `AnnealingResult` containing the best state, final state, and
    /// detailed statistics about the annealing process.
    ///
    /// # Examples
    ///
    /// ```
    /// use frostfire::prelude::*;
    ///
    /// // Implementation details omitted for brevity
    /// # #[derive(Clone)]
    /// # struct MyState;
    /// # impl State for MyState {
    /// #     fn neighbor(&self, _: &mut impl rand::Rng) -> Self { self.clone() }
    /// # }
    /// # struct MyEnergy;
    /// # impl Energy for MyEnergy {
    /// #     type State = MyState;
    /// #     fn cost(&self, _: &Self::State) -> f64 { 0.0 }
    /// # }
    ///
    /// let mut annealer = Annealer::new(
    ///     MyState,
    ///     MyEnergy,
    ///     GeometricSchedule::new(100.0, 0.95),
    ///     seeded_rng(42),
    ///     10000,
    /// );
    ///
    /// let result = annealer.run_with_stats();
    /// println!("Best energy: {}", result.best_energy);
    /// println!("Acceptance ratio: {}", result.accepted_moves as f64 / result.iterations as f64);
    /// ```
    pub fn run_with_stats(&mut self) -> AnnealingResult<S> {
        // Initialize variables
        let initial_temp = self.schedule.initial_temp();
        let mut current_temp = initial_temp;
        let mut current_energy = self.energy.cost(&self.state);
        
        // Save the initial state as the best state
        self.best_state = Some(self.state.clone());
        self.best_energy = current_energy;
        
        // Reset statistics
        self.accepted_moves = 0;
        self.rejected_moves = 0;
        
        // Main annealing loop
        for i in 0..self.max_iters {
            // Generate a neighboring state
            let new_state = self.state.neighbor(&mut self.rng);
            let new_energy = self.energy.cost(&new_state);
            
            // Calculate the energy difference
            let delta = new_energy - current_energy;
            
            // Decide whether to accept the new state
            if transition::accept(delta, current_temp, &mut self.rng) {
                // Accept the new state
                self.state = new_state;
                current_energy = new_energy;
                
                // Update statistics
                self.accepted_moves += 1;
                
                // Update the best state if we found a better one
                if new_energy < self.best_energy {
                    self.best_state = Some(self.state.clone());
                    self.best_energy = new_energy;
                }
            } else {
                // Reject the new state
                self.rejected_moves += 1;
            }
            
            // Update the temperature according to the cooling schedule
            current_temp = self.schedule.next_temp(current_temp, i);
        }
        
        // Create the result object
        AnnealingResult {
            best_state: self.best_state.as_ref().unwrap().clone(),
            best_energy: self.best_energy,
            final_state: self.state.clone(),
            final_energy: current_energy,
            iterations: self.max_iters,
            accepted_moves: self.accepted_moves,
            rejected_moves: self.rejected_moves,
            initial_temp,
            final_temp: current_temp,
        }
    }
}
