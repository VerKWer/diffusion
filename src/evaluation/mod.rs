use std::fmt::Display;

use rand::Rng;

use crate::{diffusion::DiffusionFunc, globals::N_SAMPLES};

pub mod wasserstein_arith;
pub mod wasserstein_geom;
mod bitflips;
mod avalanche;

pub trait Evaluator<F: DiffusionFunc>: Display {
	fn new(func: F) -> Self;

	fn random(rng: &mut impl Rng) -> Self;

    fn get_age(&self) -> u32;

    fn get_loss(&self) -> f32;

	fn get_func(&self) -> &F;

    fn update(&mut self, samples: &[u64; (N_SAMPLES) as usize]) -> f32;

}
