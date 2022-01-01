use std::{fmt::{self, Display, Formatter}, mem};

use rand::Rng;

use super::{
	shifts::{MAX_S1, MIN_S2},
	DiffusionFunc,
};


#[derive(Default, Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct MRXSM {
	pub m1: u64,
	pub m2: u64,
	pub s1: u32,
	pub s2: u32,
}

impl MRXSM {
	#[inline(always)]
	pub fn new(m1: u64, m2: u64, s1: u32, s2: u32) -> Self {
		let s1 = s1.min(MAX_S1);
		let s2 = s2.max(MIN_S2[s1 as usize]);
		MRXSM { m1, m2, s1, s2 }
	}
}

impl Display for MRXSM {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		write!(f, "MRXSM{{")?;
		write!(f, "m1:{:#x}, ", self.m1)?;
		write!(f, "m2:{:#x}, ", self.m2)?;
		write!(f, "s1:{}, ", self.s1)?;
		write!(f, "s2:{}", self.s2)?;
		write!(f, "}}")
	}
}

impl DiffusionFunc for MRXSM {
	#[inline(always)]
	fn diffuse(&self, mut x: u64) -> u64 {
		x = x.overflowing_mul(self.m1).0;
		// Generates many more instructions
		// x ^= x.checked_shr(s1 + (x.checked_shr(s2).unwrap_or(0) as u32)).unwrap_or(0);
		// x ^= x >> (s1 + ((x >> s2) as u32));

		// This is the default (scalar) behaviour on x86 (shifting by 64 is the same as not shifting at all)
		x ^= x.wrapping_shr(self.s1 + x.wrapping_shr(self.s2) as u32);
		x = x.overflowing_mul(self.m2).0;
		x
	}

	#[inline(always)]
	fn random(rng: &mut impl Rng) -> Self {
		// Note: Expected number of set bits in a random integer is half.
		let m1 = rng.gen::<u64>() | 1;
		let m2 = rng.gen::<u64>() | 1;
		let s1: u32 = rng.gen_range(0..=MAX_S1);
		let s2: u32 = rng.gen_range(MIN_S2[s1 as usize]..=63);
		Self::new(m1, m2, s1, s2)
	}

	#[inline(always)]
	fn crossover(&self, other: &Self, rng: &mut impl Rng) -> [Self; 2] {
		let mut result = [MRXSM::default(), MRXSM::default()];
		let m1s = super::crossover(self.m1, other.m1, rng);
		let m2s = super::crossover(self.m2, other.m2, rng);
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
			let m1 = super::mutate(m1s[i], rng) | 1;
			let m2 = super::mutate(m2s[i], rng) | 1;
			result[i] = Self::new(m1, m2, s1, s2);
		}
		result
	}
}
