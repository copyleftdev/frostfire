//! Core components of the frostfire simulated annealing library.
//!
//! This module contains the fundamental abstractions and implementations
//! that form the backbone of the frostfire library:
//!
//! - `annealer`: The main optimization engine
//! - `state`: The representation of candidate solutions
//! - `energy`: The cost function to be minimized
//! - `transition`: Acceptance criteria for proposed state transitions
//! - `schedule`: Cooling schedules that control the annealing process

pub mod annealer;
pub mod state;
pub mod energy;
pub mod transition;
pub mod schedule;
