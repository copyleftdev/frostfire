# Frostfire

A modular, mathematically rigorous, performant, reusable simulated annealing optimization engine implemented in Rust.

## Features

- **Correct-by-construction**: Mathematical rigor backed by extensive test suites
- **Modularity**: Plug-and-play across any domain requiring heuristic optimization
- **Determinism**: Fully reproducible runs, enhancing debugging and scientific rigor
- **Performance**: Zero-cost abstractions enabling efficient runtime behavior

## Usage

```rust
use frostfire::prelude::*;

// Define your problem state
#[derive(Clone)]
struct MyState { /* ... */ }

impl State for MyState {
    fn neighbor(&self, rng: &mut impl Rng) -> Self {
        // Generate a neighbor state
    }
}

// Define your energy/cost function
struct MyEnergy { /* ... */ }

impl Energy for MyEnergy {
    type State = MyState;
    
    fn cost(&self, state: &Self::State) -> f64 {
        // Calculate cost
    }
}

// Create and run the annealer
fn main() {
    let initial_state = MyState { /* ... */ };
    let energy = MyEnergy { /* ... */ };
    let schedule = GeometricSchedule::new(100.0, 0.95);
    
    let mut annealer = Annealer::new(
        initial_state,
        energy,
        schedule,
        seeded_rng(42), // For reproducibility
        10000, // Max iterations
    );
    
    let (best_state, best_energy) = annealer.run();
    println!("Best energy: {}", best_energy);
}
```

## Applications

- Machine learning hyperparameter optimization
- Operations research problems
- Real-time optimization
- Simulation parameter tuning
- Embedded systems optimization

## Documentation

For full API documentation, please visit [docs.rs/frostfire](https://docs.rs/frostfire).

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
