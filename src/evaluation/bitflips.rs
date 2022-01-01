// use std::arch::x86_64::{_mm256_set1_epi64x, _mm256_xor_epi64};

use aligned_array::{A32, Aligned};

use crate::{diffusion::DiffusionFunc, globals::N_SAMPLES_PER_ROUND, utils::wasserstein};

#[derive(Debug, PartialEq, Eq)]
// pub struct Bitflips(Aligned<A32, [[u32; 68]; 64]>);
pub struct Bitflips([[u32; 65]; 64]);

const BITS: Aligned<A32, [u64; 64]> = init_bits();
const fn init_bits() -> Aligned<A32, [u64; 64]> {
	let mut result = [1_u64; 64];
	let mut shift = 0;
	while shift < 64 {
		result[shift] <<= shift;
		shift += 1;
	}
	Aligned(result)
}

impl Bitflips {

	pub fn of(f: &impl DiffusionFunc, samples: &[u64; N_SAMPLES_PER_ROUND as usize]) -> Self {
		// debug_assert!(samples.len() == N_SAMPLES_PER_ROUND);
		// let mut n_flips: Aligned<A32, _> = Aligned([[0_u32; 68]; 64]);
		let mut n_flips = [[0_u32; 65]; 64];
		for &x in samples {
			let h = f.diffuse(x);
			#[allow(clippy::needless_range_loop)]
			for shift in 0..64 {
				let diff = h ^ f.diffuse(x ^ BITS[shift]);
				n_flips[shift][diff.count_ones() as usize] += 1;

			}

			// let xs = unsafe { _mm256_set1_epi64x(x) };
			// let h = unsafe { _mm256_set1_epi64x(f.diffuse(x) as i64) };
			// #[allow(clippy::needless_range_loop)]
			// for i in (0..64).step_by(4) {
			// 	let bits = unsafe { utils::read_m256i(BITS.as_ptr().offset(i)) };
			// 	let flipped = unsafe { _mm256_xor_epi64(xs, bits) };
			// 	let h2 = f.diffuse4(flipped);
			// 	let diff = unsafe { _mm256_xor_epi64(h, h2) };
			// 	n_flips[shift][diff.count_ones() as usize] += 1;

			// }
		}
		Self(n_flips)
	}

	pub fn w1s(&self) -> [f32; 64] {
		let mut w1s = [0_f32; 64];
		for (counts, w1) in self.0.iter().zip(w1s.iter_mut()) {
			// *w1 = wasserstein::of_counts(&counts[..65].try_into().unwrap());
			*w1 = wasserstein::of_counts(counts);
		}
		w1s
	}
}


#[cfg(test)]
mod tests {
	use rand::Rng;

	use crate::diffusion::rxsm::RXSM;
	use super::*;

	#[test]
	fn test_rxsm_example() {
		let f = RXSM::new(0xa4001226aaaaaab, 21, 59);
		let mut rng = rand::thread_rng();
		let mut samples = [0_u64; N_SAMPLES_PER_ROUND as usize];
		for i in 0..N_SAMPLES_PER_ROUND {
			samples[i as usize] = rng.gen();
		}
		let n_flips = Bitflips::of(&f, &samples);
		let w1s = n_flips.w1s();
		dbg!("{}", w1s);
		let mut max_idx = 0;
		let mut max_w1 = 0.0;
		for (i, &w1) in w1s.iter().enumerate() {
			if w1 > max_w1 {
				max_w1 = w1;
				max_idx = i;
			}
		}
		println!("max w1: {}, max idx: {}", max_w1, max_idx);
		print!("[");
		for f in n_flips.0[max_idx] {
			print!("{}, ", f);
		}
		println!("]");
	}
}
