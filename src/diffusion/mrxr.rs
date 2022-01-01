use std::{fmt::{self, Display, Formatter}, mem};

use rand::Rng;

use super::{
	shifts::{MAX_S1, MIN_S2},
	DiffusionFunc,
};


#[derive(Default, Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct MRXR {
	pub m: u64,
	pub s1: u32,
	pub s2: u32,
}

impl MRXR {
	#[inline(always)]
	pub fn new(m: u64, s1: u32, s2: u32) -> Self {
		let s1 = s1.min(MAX_S1);
		let s2 = s2.max(MIN_S2[s1 as usize]);
		MRXR { m, s1, s2 }
	}
}

impl Display for MRXR {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		write!(f, "MRXR{{")?;
		write!(f, "m:{:#x}, ", self.m)?;
		write!(f, "s1:{}, ", self.s1)?;
		write!(f, "s2:{}", self.s2)?;
		write!(f, "}}")
	}
}

impl DiffusionFunc for MRXR {
	#[inline(always)]
	fn diffuse(&self, mut x: u64) -> u64 {
		x = x.overflowing_mul(self.m).0;
		let s = self.s1 + x.wrapping_shr(self.s2) as u32;
		x ^= x.rotate_right(s);
		x
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
		let mut result = [MRXR::default(), MRXR::default()];
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
