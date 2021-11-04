use rand::Rng;

use crate::globals::{N_ROUNDS, N_SAMPLES};

pub trait Endo64 {
    fn diffuse(&self, x: u64) -> u64;
}

pub trait DiffusionFunc: Endo64 + Sized {
    fn random(rng: &mut impl Rng) -> Self;

    fn get_age(&self) -> u32;

    fn get_loss(&self) -> f32;

    fn update(&mut self, samples: &[u64; (N_SAMPLES * N_ROUNDS) as usize]) -> f32;

    fn crossover(&self, other: &Self, rng: &mut impl Rng) -> [Self; 2];
}
