use std::{
	fmt::{self, Display, Formatter},
	mem,
};

use rand::Rng;

use super::DiffusionFunc;


#[derive(Default, Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct MXR {
	pub m: u64,
	pub s: u32,
}

impl MXR {
	#[inline(always)]
	pub fn new(m: u64, s: u32) -> Self {
		let s = s.min(63);
		MXR { m, s }
	}
}

impl Display for MXR {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		write!(f, "MROR{{")?;
		write!(f, "m:{:#x}, ", self.m)?;
		write!(f, "s:{}", self.s)?;
		write!(f, "}}")
	}
}

impl DiffusionFunc for MXR {
	#[inline(always)]
	fn diffuse(&self, mut x: u64) -> u64 {
		x = x.overflowing_mul(self.m).0;
		x ^= x.rotate_right(self.s);
		x
	}
	#[inline(always)]
	fn random(rng: &mut impl Rng) -> Self {
		// Note: Expected number of set bits in a random integer is half.
		let m = rng.gen::<u64>() | 1;
		let s: u32 = rng.gen_range(0..64);
		Self::new(m, s)
	}

	#[inline(always)]
	fn crossover(&self, other: &Self, rng: &mut impl Rng) -> [Self; 2] {
		let mut result = [MXR::default(), MXR::default()];
		let ms = super::crossover(self.m, other.m, rng);
		for i in 0..2 {
			let (mut s_min, mut s_max) = (self.s, other.s);
			if s_min > s_max {
				mem::swap(&mut s_min, &mut s_max);
			}
			let m = super::mutate(ms[i], rng) | 1;
			let s = rng.gen_range(s_min..=s_max);
			result[i] = Self::new(m, s);
		}
		result
	}
}
