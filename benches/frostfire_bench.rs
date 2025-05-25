//! Benchmarks for the frostfire simulated annealing library.
//!
//! This module provides reproducible performance benchmarks for various
//! optimization problems solved using the frostfire library.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use frostfire::prelude::*;
use rand::Rng;

// TSP Benchmarking

#[derive(Clone)]
struct TspProblem {
    cities: Vec<(f64, f64)>,
}

impl TspProblem {
    fn random(num_cities: usize, rng: &mut impl Rng) -> Self {
        let cities = (0..num_cities)
            .map(|_| (rng.gen_range(0.0..100.0), rng.gen_range(0.0..100.0)))
            .collect();
        
        Self { cities }
    }
    
    fn distance(&self, city1: usize, city2: usize) -> f64 {
        let (x1, y1) = self.cities[city1];
        let (x2, y2) = self.cities[city2];
        
        ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt()
    }
    
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

#[derive(Clone)]
struct TspState {
    tour: Vec<usize>,
}

impl TspState {
    fn random(num_cities: usize, rng: &mut impl Rng) -> Self {
        let mut tour: Vec<usize> = (0..num_cities).collect();
        for i in (1..num_cities).rev() {
            let j = rng.gen_range(0..=i);
            tour.swap(i, j);
        }
        
        Self { tour }
    }
}

impl State for TspState {
    fn neighbor(&self, rng: &mut impl Rng) -> Self {
        let mut new_tour = self.tour.clone();
        let idx1 = rng.gen_range(0..new_tour.len());
        let idx2 = rng.gen_range(0..new_tour.len());
        
        if idx1 != idx2 {
            new_tour.swap(idx1, idx2);
        }
        
        Self { tour: new_tour }
    }
}

struct TspEnergy {
    problem: TspProblem,
}

impl Energy for TspEnergy {
    type State = TspState;
    
    fn cost(&self, state: &Self::State) -> f64 {
        self.problem.tour_distance(&state.tour)
    }
}

// Rastrigin Benchmarking

#[derive(Clone)]
struct RastriginState {
    coords: Vec<f64>,
    range: (f64, f64),
}

impl RastriginState {
    fn new(dimensions: usize, range: (f64, f64), rng: &mut impl Rng) -> Self {
        let coords = (0..dimensions)
            .map(|_| rng.gen_range(range.0..range.1))
            .collect();
            
        Self { coords, range }
    }
}

impl State for RastriginState {
    fn neighbor(&self, rng: &mut impl Rng) -> Self {
        let mut new_coords = self.coords.clone();
        
        let idx = rng.gen_range(0..new_coords.len());
        let perturbation = rng.gen_range(-0.1..0.1);
        new_coords[idx] += perturbation;
        
        new_coords[idx] = new_coords[idx].max(self.range.0).min(self.range.1);
        
        Self {
            coords: new_coords,
            range: self.range,
        }
    }
}

struct RastriginEnergy;

impl Energy for RastriginEnergy {
    type State = RastriginState;
    
    fn cost(&self, state: &Self::State) -> f64 {
        use std::f64::consts::PI;
        
        let n = state.coords.len() as f64;
        
        let sum: f64 = state.coords.iter()
            .map(|&x| x * x - 10.0 * (2.0 * PI * x).cos())
            .sum();
            
        10.0 * n + sum
    }
}

// Knapsack Benchmarking

#[derive(Clone)]
struct Item {
    weight: f64,
    value: f64,
}

#[derive(Clone)]
struct KnapsackProblem {
    items: Vec<Item>,
    capacity: f64,
}

impl KnapsackProblem {
    fn random(num_items: usize, capacity: f64, rng: &mut impl Rng) -> Self {
        let items = (0..num_items)
            .map(|_| Item {
                weight: rng.gen_range(1.0..20.0),
                value: rng.gen_range(1.0..50.0),
            })
            .collect();
            
        Self {
            items,
            capacity,
        }
    }
    
    fn total_weight(&self, selection: &[bool]) -> f64 {
        selection.iter()
            .zip(self.items.iter())
            .filter(|&(selected, _)| *selected)
            .map(|(_, item)| item.weight)
            .sum()
    }
    
    fn total_value(&self, selection: &[bool]) -> f64 {
        selection.iter()
            .zip(self.items.iter())
            .filter(|&(selected, _)| *selected)
            .map(|(_, item)| item.value)
            .sum()
    }
}

#[derive(Clone)]
struct KnapsackState {
    selection: Vec<bool>,
}

impl KnapsackState {
    fn random(num_items: usize, rng: &mut impl Rng) -> Self {
        let selection = (0..num_items)
            .map(|_| rng.gen_bool(0.5))
            .collect();
            
        Self { selection }
    }
}

impl State for KnapsackState {
    fn neighbor(&self, rng: &mut impl Rng) -> Self {
        let mut new_selection = self.selection.clone();
        
        let idx = rng.gen_range(0..new_selection.len());
        new_selection[idx] = !new_selection[idx];
        
        Self { selection: new_selection }
    }
}

struct KnapsackEnergy {
    problem: KnapsackProblem,
    penalty_factor: f64,
}

impl Energy for KnapsackEnergy {
    type State = KnapsackState;
    
    fn cost(&self, state: &Self::State) -> f64 {
        let total_value = self.problem.total_value(&state.selection);
        let total_weight = self.problem.total_weight(&state.selection);
        
        let base_cost = -total_value;
        
        if total_weight > self.problem.capacity {
            let violation = total_weight - self.problem.capacity;
            base_cost + self.penalty_factor * violation
        } else {
            base_cost
        }
    }
}

// Benchmark Functions

fn bench_tsp(c: &mut Criterion) {
    let mut group = c.benchmark_group("TSP");
    
    for size in [10, 20, 50] {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_batched(
                || {
                    // Setup
                    let mut rng = seeded_rng(42);
                    let problem = TspProblem::random(size, &mut rng);
                    let energy = TspEnergy { problem: problem.clone() };
                    let initial_state = TspState::random(size, &mut rng);
                    let schedule = GeometricSchedule::new(100.0, 0.95);
                    
                    Annealer::new(
                        initial_state,
                        energy,
                        schedule,
                        seeded_rng(42),
                        1000,
                    )
                },
                |mut annealer| {
                    // Benchmark
                    black_box(annealer.run())
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    
    group.finish();
}

fn bench_rastrigin(c: &mut Criterion) {
    let mut group = c.benchmark_group("Rastrigin");
    
    for dims in [2, 5, 10].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(dims), dims, |b, &dims| {
            b.iter_batched(
                || {
                    // Setup
                    let mut rng = seeded_rng(1337);
                    let range = (-5.12, 5.12);
                    let initial_state = RastriginState::new(dims, range, &mut rng);
                    let energy = RastriginEnergy;
                    let schedule = GeometricSchedule::new(10.0, 0.95);
                    
                    Annealer::new(
                        initial_state,
                        energy,
                        schedule,
                        seeded_rng(1337),
                        1000,
                    )
                },
                |mut annealer| {
                    // Benchmark
                    black_box(annealer.run())
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    
    // Compare different cooling schedules
    for schedule_type in ["geometric", "logarithmic", "adaptive"].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(*schedule_type), schedule_type, |b, schedule_type| {
            b.iter_batched(
                || {
                    // Setup
                    let mut rng = seeded_rng(1337);
                    let range = (-5.12, 5.12);
                    let initial_state = RastriginState::new(5, range, &mut rng);
                    let energy = RastriginEnergy;
                    
                    // Use a type-erased approach with runtime branching instead of static dispatch
                    // This avoids type compatibility issues with match arms
                    let max_iters = 1000;
                    let rng = seeded_rng(1337);
                    
                    if *schedule_type == "geometric" {
                        let schedule = GeometricSchedule::new(10.0, 0.95);
                        Annealer::new(
                            initial_state,
                            energy,
                            schedule,
                            rng,
                            max_iters,
                        )
                    } else if *schedule_type == "logarithmic" {
                        let schedule = LogarithmicSchedule::new(10.0);
                        Annealer::new(
                            initial_state,
                            energy,
                            schedule,
                            rng,
                            max_iters,
                        )
                    } else if *schedule_type == "adaptive" {
                        let schedule = AdaptiveSchedule::new(10.0);
                        Annealer::new(
                            initial_state,
                            energy,
                            schedule,
                            rng,
                            max_iters,
                        )
                    } else {
                        unreachable!()
                    }
                },
                |mut annealer| {
                    // Benchmark
                    black_box(annealer.run())
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    
    group.finish();
}

fn bench_knapsack(c: &mut Criterion) {
    let mut group = c.benchmark_group("Knapsack");
    
    for size in [20, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_batched(
                || {
                    // Setup
                    let mut rng = seeded_rng(777);
                    let capacity = 3.0 * size as f64;
                    let problem = KnapsackProblem::random(size, capacity, &mut rng);
                    let energy = KnapsackEnergy { 
                        problem: problem.clone(),
                        penalty_factor: 100.0,
                    };
                    let initial_state = KnapsackState::random(size, &mut rng);
                    let schedule = GeometricSchedule::new(100.0, 0.95);
                    
                    Annealer::new(
                        initial_state,
                        energy,
                        schedule,
                        seeded_rng(777),
                        1000,
                    )
                },
                |mut annealer| {
                    // Benchmark
                    black_box(annealer.run())
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_tsp,
    bench_knapsack,
    bench_rastrigin
);
criterion_main!(benches);
