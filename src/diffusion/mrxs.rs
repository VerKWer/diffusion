use std::{
	arch::x86_64::{__m256i, _mm256_add_epi64, _mm256_set1_epi64x, _mm256_srlv_epi64, _mm256_xor_epi64},
	fmt::{self, Display, Formatter},
	mem,
};

use rand::Rng;

use crate::utils;

use super::{
	shifts::{MAX_S1, MIN_S2},
	DiffusionFunc,
};


#[derive(Default, Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct MRXS {
	pub m: u64,
	pub s1: u32,
	pub s2: u32,
}

impl MRXS {
	#[inline(always)]
	pub fn new(m: u64, s1: u32, s2: u32) -> Self {
		let s1 = s1.min(MAX_S1);
		let s2 = s2.max(MIN_S2[s1 as usize]);
		MRXS { m, s1, s2 }
	}
}

impl Display for MRXS {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		write!(f, "MRXS{{")?;
		write!(f, "m:{:#x}, ", self.m)?;
		write!(f, "s1:{}, ", self.s1)?;
		write!(f, "s2:{}", self.s2)?;
		write!(f, "}}")
	}
}

impl DiffusionFunc for MRXS {
	#[inline(always)]
	fn diffuse(&self, mut x: u64) -> u64 {
		x = x.overflowing_mul(self.m).0;
		x ^= x.wrapping_shr(self.s1 + x.wrapping_shr(self.s2) as u32);
		x
	}

	#[inline(always)]
	fn diffuse4(&self, mut xs: __m256i) -> __m256i {
		unsafe {
			let m = _mm256_set1_epi64x(self.m as i64);
			let s1 = _mm256_set1_epi64x(self.s1 as i64);
			let s2 = _mm256_set1_epi64x(self.s2 as i64);
			xs = utils::mul_m256i64(xs, m);
			let s = _mm256_add_epi64(s1, _mm256_srlv_epi64(xs, s2));
			xs = _mm256_xor_epi64(xs, _mm256_srlv_epi64(xs, s));
		}
		xs
	}

	#[inline(always)]
	fn random(rng: &mut impl Rng) -> Self {
		// Note: Expected number of set bits in a random integer is half.
		let m = rng.gen::<u64>() | 1;
		let s1: u32 = rng.gen_range(0..=MAX_S1);
		let s2: u32 = rng.gen_range(MIN_S2[s1 as usize]..=63);
		Self::new(m, s1, s2)
	}

	#[inline(always)]
	fn crossover(&self, other: &Self, rng: &mut impl Rng) -> [Self; 2] {
		let mut result = [MRXS::default(), MRXS::default()];
		let ms = super::crossover(self.m, other.m, rng);
		for i in 0..2 {
			let (mut s_min, mut s_max) = (self.s1, other.s1);
			if s_min > s_max {
				mem::swap(&mut s_min, &mut s_max);
			}
			let s1: u32 = rng.gen_range(s_min..=s_max);
			let (mut s_min, mut s_max) = (self.s2, other.s2);
			if s_min > s_max {
				mem::swap(&mut s_min, &mut s_max);
			}
			let s2: u32 = rng.gen_range(0.max(s_min - 1)..=63.min(s_max + 1));
			let m = super::mutate(ms[i], rng) | 1;
			result[i] = Self::new(m, s1, s2);
		}
		result
	}
}


#[cfg(test)]
mod tests {
	use aligned_array::{Aligned, A32};

	use super::*;

	#[test]
	fn test_diffuse4() {
		let mut rng = rand::thread_rng();
		for _ in 0..100 {
			let f = MRXS::random(&mut rng);
			let xs: Aligned<A32, [u64; 4]> = Aligned(rng.gen());
			let ds = f.diffuse4(unsafe { mem::transmute::<Aligned<A32, [u64; 4]>, __m256i>(xs.clone()) });
			let ds = unsafe { mem::transmute::<__m256i, [u64; 4]>(ds) };
			for (i, &x) in xs.iter().enumerate() {
				assert_eq!(f.diffuse(x), ds[i]);
			}
		}
	}
}
