use std::{arch::x86_64::__m256i, fmt::Display, mem};

use aligned_array::{A32, Aligned};
use rand::Rng;

use crate::globals::{CROSSOVER_BITS, MUTATION_ODDS};

mod shifts;
pub mod mrxsm;
pub mod mrxs;
pub mod rxsm;
pub mod mxr;
pub mod mrxr;

pub trait DiffusionFunc: Sized + Display {
    fn diffuse(&self, x: u64) -> u64;

	fn diffuse4(&self, xs: __m256i) -> __m256i {
		let xs = unsafe { mem::transmute::<__m256i, [u64; 4]>(xs) };
		let mut ys: Aligned<A32, _> = Aligned([0_u64; 4]);
		for (x, y) in xs.into_iter().zip(ys.iter_mut()) {
			*y = self.diffuse(x);
		}
		unsafe { mem::transmute::<Aligned<A32, [u64; 4]>, __m256i>(ys) }
	}

    fn random(rng: &mut impl Rng) -> Self;

    fn crossover(&self, other: &Self, rng: &mut impl Rng) -> [Self; 2];
}

fn crossover(parent1: u64, parent2: u64, rng: &mut impl Rng) -> [u64; 2] {
	let mask = 1_u64.wrapping_shl(CROSSOVER_BITS) - 1;
	// println!("Crossover mask: {:#x}", mask);
	let mask = mask.rotate_right(rng.gen());
	let inv_mask = !mask;
	let child1 = (parent1 & mask) | (parent2 & inv_mask);
	let child2 = (parent1 & inv_mask) | (parent2 & mask);
	[child1, child2]
}

/** Each bit is flipped with a 1/odds probability. */
fn mutate(x: u64, rng: &mut impl Rng) -> u64 {
	let mut mask = 0_u64;
	let mut bit = 0x8000_0000_0000_0000_u64;
	for _ in 0..64 {
		if rng.gen_range(0..MUTATION_ODDS) == 0_u32 {
			mask |= bit
		}
		bit >>= 1;
	}
	// println!("Mutating {} bits (expected: {:.2}, mask: {:#x})", mask.count_ones(), 64.0/m as f32, mask);
	x ^ (x & mask)
}
