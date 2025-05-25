//! Convenient re-exports of commonly used types and functions.
//!
//! This module re-exports the most commonly used items from the frostfire crate,
//! allowing users to import them all with a single `use frostfire::prelude::*` statement.

pub use crate::core::annealer::{Annealer, AnnealingResult};
pub use crate::core::energy::Energy;
pub use crate::core::schedule::{
    AdaptiveSchedule, GeometricSchedule, LogarithmicSchedule, Schedule,
};
pub use crate::core::state::State;
pub use crate::core::transition::accept;
pub use crate::rng::seeded_rng::seeded_rng;

// Re-export commonly used external types
pub use rand::rngs::StdRng;
pub use rand::Rng;
