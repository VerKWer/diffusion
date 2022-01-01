#![allow(unused_imports)]
use crate::{diffusion::{mrxr::MRXR, mrxs::MRXS, mrxsm::MRXSM, mxr::MXR, rxsm::RXSM}, evaluation::wasserstein_arith::WassersteinArith};

pub type F = MRXSM;  // the type of diffusion function
pub type E = WassersteinArith<F>;  // the type of evaluation strategy used

#[cfg(feature="profile")]
pub const N_GENERATIONS: u32 = 1;
#[cfg(not(feature="profile"))]
pub const N_GENERATIONS: u32 = 100;

cfg_if! {
	if #[cfg(debug)] {
		pub const N_THREADS: u32 = 1;
		pub const GENERATION_SIZE: u32 = 32;
		pub const ELITISM: u32 = 4;
		pub const N_SAMPLES: u32 = 100;
		/** Used for Wasserstein evaluators. Must be a divisor of `N_SAMPLES`. */
		pub const N_ROUNDS: u32 = 1;
	} else {
		pub const N_THREADS: u32 = 8;
		pub const GENERATION_SIZE: u32 = 512;
		pub const ELITISM: u32 = 50;
		pub const N_SAMPLES: u32 = 100000;
		/** Used for Wasserstein evaluators. Must be a divisor of `N_SAMPLES`. */
		pub const N_ROUNDS: u32 = 100;
	}
}
const_assert_eq!(0, N_SAMPLES % N_ROUNDS);
pub const N_SAMPLES_PER_ROUND: u32 = N_SAMPLES / N_ROUNDS;
const_assert!(GENERATION_SIZE - ELITISM > 0 && (GENERATION_SIZE - ELITISM) & 1 == 0);

pub const TOURNAMENT_SIZE: u32 = 4;
pub const CROSSOVER_BITS: u32 = 32;
const_assert!(CROSSOVER_BITS > 0 && CROSSOVER_BITS < 33);
pub const MUTATION_ODDS: u32 = 8;

/** Number of generations after which we exchange the samples. Must be a power of 2. */
pub const SAMPLE_LIFETIME: u32 = 8;
const_assert_eq!(1, SAMPLE_LIFETIME.count_ones());
