#![allow(dead_code)]
use aligned_array::{Aligned, A32};
use std::{arch::x86_64::*, fmt::Display, mem};

use crate::{diffusion::DiffusionFunc, utils::{self, bitset}};


/** OBS: These are mirrored from how they apear in our Jupyter notebook because that's more cache efficient. */
#[derive(Debug, PartialEq, Eq)]
pub struct AvalancheDiagram {
	expected: u32,
	vals: Aligned<A32, [[u32; 64]; 64]>,
}

impl Display for AvalancheDiagram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		writeln!(f, "AvalancheDiagram{{vals:")?;
        write!(f, "[")?;
		for i in 0..64 {
			let row = &self.vals[i];
			write!(f, "[")?;
			#[allow(clippy::needless_range_loop)]
			for j in 0..64 {
				let x = row[j];
				write!(f, "{}", x)?;
				if j != 63 { write!(f, ",")?; }
			}
			if i == 63 { writeln!(f, "]]")?; }
			else { writeln!(f, "],")?; }
		}

		write!(f, "expected: {}}}", self.expected)
    }
}


impl AvalancheDiagram {
	pub fn new(n_samples: u32, vals: Aligned<A32, [[u32; 64]; 64]>) -> Self {
		#[cfg(debug_assertions)]
		{
			assert!(n_samples & 1 == 0); // only allow even number of samples
			for row in vals.iter() {
				for &val in row {
					assert!(val <= n_samples);
				}
			}
		}
		Self { expected: n_samples >> 1, vals }
	}

	pub fn of(f: &impl DiffusionFunc, samples: &[u64]) -> Self {
		let mut vals: Aligned<A32, _> = Aligned([[0_u32; 64]; 64]);
		for &x in samples {
			let h = f.diffuse(x);
			for shift in 0..64 {
				let row = &mut vals[shift];
				let mut diff = (h ^ f.diffuse(x ^ (1 << shift))) as u64;
				// Very slow
				// loop {
				//     let first_set = diff.trailing_zeros();
				//     if first_set == 64 { break; }
				//     row[first_set as usize] += 1;
				//     diff ^= 1 << first_set;
				// }
				// Slow
				// while diff != 0 {
				// 	let least_set_bit = diff & diff.wrapping_neg();
				// 	let idx = least_set_bit.trailing_zeros() as usize;
				// 	row[idx] += 1;
				// 	diff ^= least_set_bit;
				// }
                // Fast
                for k in (0..64).step_by(16) {
                    let byte1 = diff as u8;
                    let byte2 = (diff >> 8) as u8;
                    diff >>= 16;
                    unsafe {
                        let bits1 = mem::transmute::<[u32; 8], __m256i>(*bitset::get_set_bits(byte1));
                        let bits2 = mem::transmute::<[u32; 8], __m256i>(*bitset::get_set_bits(byte2));
                        let ptr1 = row.as_mut_ptr().offset(k);
                        let ptr2 = ptr1.offset(8);
                        let prev1 = utils::read_m256i(ptr1);
                        let prev2 = utils::read_m256i(ptr2);
                        utils::write_m256i(_mm256_add_epi32(prev1, bits1), ptr1);
                        utils::write_m256i(_mm256_add_epi32(prev2, bits2), ptr2);
                    }
                }
			}
		}
		Self::new(samples.len() as u32, vals)
	}

	pub fn sse_reference(&self) -> f32 {
		debug_assert!(self.expected < 1024);
		let mut result = 0_f32;
		for row in self.vals.iter() {
			for val in row {
				let err = val.wrapping_sub(self.expected);
				let err = err.wrapping_mul(err);
				result += err as f32/self.expected as f32;
			}
		}
		result
	}

	pub fn sse(&self) -> f32 {
		debug_assert!(self.expected < 1024);
		unsafe {
			let exp = _mm256_set1_epi32(self.expected as i32);
			let mut accum = _mm256_set1_epi32(0);
			/* I tried unrolling the inner loop, adding a second accumulator and having one big loop instead of nesting
			them. None of it seems to make any difference. */
			for i in 0..64 {
				let row = &self.vals[i];
				let mut row_ptr = row.as_ptr();
				for _ in 0..8 {
					// let vals = _mm256_load_epi32(row_ptr as *const i32);  // tanks performance
					let vals = utils::read_m256i(row_ptr); // much faster
					let err = _mm256_sub_epi32(vals, exp);
					// NOTE: _mm256_mul_epi32 operates on low 32 bits of 4 64-bit ints; so that's not what we want.
					let err = _mm256_mullo_epi32(err, err);
					accum = _mm256_add_epi32(accum, err);
					row_ptr = row_ptr.offset(8);
				}
			}
			// Note: self.expected < 2**10 and f32 is exact for integers < 2**23.
			utils::hsum_m256i32(accum) as u32 as f32/self.expected as f32
		}
	}
}

#[cfg(test)]
mod tests {
	use crate::diffusion::mrxsm::MRXSM;

use super::*;
	use rand::Rng;

	#[test]
	fn worst_case_test() {
		let vals: Aligned<A32, _> = Aligned([[0_u32; 64]; 64]);
		let expected: u32 = rand::thread_rng().gen_range(2..1_000);
		let err = (4096* expected) as f32;
		let diag = AvalancheDiagram::new(expected << 1, vals);
		assert_eq!(err, diag.sse_reference());
		assert_eq!(err, diag.sse());
		let vals: Aligned<A32, _> = Aligned([[0_u32; 64]; 64]);
		let expected: u32 = 1023;
		let err = (4096* expected) as f32;
		let diag = AvalancheDiagram::new(expected << 1, vals);
		assert_eq!(err, diag.sse_reference());
		assert_eq!(err, diag.sse());
	}

	#[test]
	fn test_perfect_diagram() {
		for _ in 0..1000 {
			let expected: u32 = rand::thread_rng().gen_range(2..1_000);
			let vals: Aligned<A32, _> = Aligned([[expected; 64]; 64]);
			let diag = AvalancheDiagram::new(expected << 1, vals);
			assert_eq!(0.0, diag.sse_reference());
			assert_eq!(0.0, diag.sse());
		}
	}

	#[test]
	fn compare_small_sse() {
		let mut rng = rand::thread_rng();
		for _ in 0..10 {
			let expected: u32 = rng.gen_range(2..1_000);
			let n_samples = expected << 1;
			let mut vals: Aligned<A32, _> = Aligned([[0_u32; 64]; 64]);
			for i in 0..64 {
				for j in 0..64 {
					vals[i][j] = rng.gen_range(0..=n_samples);
				}
			}
			let diag = AvalancheDiagram::new(n_samples, vals);
			let err = diag.sse_reference();
			assert!((err - diag.sse()).abs()/err < 0.001);
		}
	}

	#[test]
	fn avalanche_diagram_of_func() {
		let f = MRXSM::new(0x6eed0e9da4d94a4f, 0x6eed0e9da4d94a4f, 32, 60);
		let mut rng = rand::thread_rng();
		const N_SAMPLES: usize = 1000;
		let mut samples = [0_u64; N_SAMPLES];
		for sample in samples.iter_mut() {
			*sample = rng.gen();
		}
		let diag = AvalancheDiagram::of(&f, &samples);
		let sse = diag.sse_reference();
		println!("sse: {}", sse);
		assert!(sse >= 1900.0);
		assert!(sse <= 2200.0);
	}

}
