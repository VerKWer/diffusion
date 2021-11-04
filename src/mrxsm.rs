use std::fmt::{self, Display, Formatter};

use rand::Rng;

use crate::{bitflips::Bitflips, diffusion::{DiffusionFunc, Endo64}, globals::{CROSSOVER_BITS, MUTATION_ODDS, N_ROUNDS, N_SAMPLES}};


/* There is a problem with right bit-shift being inconsistent on x86. For scalar operands, both SHRX and SHRD mask
 * the count (number of bits to shift), meaning that e.g. shifting by 64 is the same as not shifting at all.
 * On the other hand, VPSRLQ (a.k.a. _mm256_srl_epi64 & _mm256_srli_epi64) returns 0 if the shift is > 63.
 * We could use Rust's checked_shr but that leads to additional instructions. Let's just restrict to a shift < 64. */
pub const MAX_TOTAL_SHIFT: u32 = 63;

/* This also means that we need to restict s1 because the total shift is s1 + (x >> s2) > s1 (we can't shift x to 0). */
pub const MAX_S1: u32 = MAX_TOTAL_SHIFT - 1;


const fn init_min_s2() -> [u32; MAX_S1 as usize + 1] {
	let mut min_s2: [u32; MAX_S1 as usize + 1] = [0; MAX_S1 as usize + 1];
	let mut s1 = 0_u32;
	while s1 <= MAX_S1 {
		let log2_floor = 31 - (MAX_TOTAL_SHIFT + 1 - s1).leading_zeros() as i32;
		min_s2[s1 as usize] = 64 - log2_floor as u32;
		s1 += 1;
	}
	min_s2
}
pub const MIN_S2: [u32; MAX_S1 as usize + 1] = init_min_s2();


#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct MRXSM {
	pub m1: u64,
	pub m2: u64,
	pub s1: u32,
	pub s2: u32,
	#[serde(with="serde_arrays")]
	w1s: [f32; 64],
	max_w1: f32,
	pub age: u32,
}

impl Display for MRXSM {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		write!(f, "DiffusionFunc{{")?;
		write!(f, "m1:{:#x}, ", self.m1)?;
		write!(f, "m2:{:#x}, ", self.m2)?;
		write!(f, "s1:{}, ", self.s1)?;
		write!(f, "s2:{}, ", self.s2)?;
		write!(f, "max_w1:{}, ", self.max_w1)?;
		write!(f, "age:{}", self.age)?;
		write!(f, "}}")
	}
}

impl Default for MRXSM {
	#[inline(always)]
	fn default() -> Self { Self { m1: 0, m2: 0, s1: 0, s2: 0, w1s: [f32::MAX; 64], max_w1: f32::MAX, age: 0 } }
}


impl MRXSM {
	#[inline]
	pub fn new(m1: u64, m2: u64, s1: u32, s2: u32) -> Self {
		let s1 = s1.min(MAX_S1);
		let s2 = s2.max(MIN_S2[s1 as usize]);
		MRXSM { m1, m2, s1, s2, w1s: [f32::MAX; 64], max_w1: f32::MAX, age: 0 }
	}

	/** Updates the stored Wasserstein distances of this diffusion function and returns the updated maximum value. */
	#[inline(always)]
	fn update_w1s(&mut self, w1s: [f32; 64]) -> f32 {
		self.age += 1;
		let mut m = 0_f32;
		if self.age == 1 {
			// this is the first SSE calculated
			self.w1s = w1s;
			for w in self.w1s {
				if w > m {
					m = w;
				}
			}
		} else {
			for i in 0..64 {
                let w1 = self.w1s[i] + w1s[i];
				self.w1s[i] = w1;
				if w1 > m {
					m = w1;
				}
			}
			m /= self.age as f32;
		}
		self.max_w1 = m;
		m
	}
}

impl Endo64 for MRXSM {
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
}

impl DiffusionFunc for MRXSM {
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
    fn get_age(&self) -> u32 { self.age }

	#[inline(always)]
	fn get_loss(&self) -> f32 { self.max_w1 }

	#[inline(always)]
	fn update(&mut self, samples: &[u64; (N_SAMPLES * N_ROUNDS) as usize]) -> f32 {
		let mut avg = [0_f32; 64];
		for round in 0..N_ROUNDS as usize {
			let samples = &samples[round..(round + N_SAMPLES as usize)];
			let w1s = Bitflips::of(self, samples).w1s();
			for i in 0..64 {
				avg[i] += w1s[i];
			}
		}
		for i in 0..64 {
			avg[i] /= N_ROUNDS as f32;
		}
		self.update_w1s(avg)
	}

	#[inline(always)]
	fn crossover(&self, other: &Self, rng: &mut impl Rng) -> [Self; 2] {
        let mut result = [MRXSM::default(), MRXSM::default()];
		let m1 = crossover(self.m1, other.m1, rng);
		let m2 = crossover(self.m2, other.m2, rng);
		for i in 0..2 {
			let (mut s_min, mut s_max) = (self.s1, other.s1);
			if s_min > s_max {
				let tmp = s_min;
				s_min = s_max;
				s_max = tmp;
			}
			let s1: u32 = rng.gen_range(s_min..=s_max);
			let (mut s_min, mut s_max) = (self.s2, other.s2);
			if s_min > s_max {
				let tmp = s_min;
				s_min = s_max;
				s_max = tmp;
			}
			let s2: u32 = rng.gen_range(0.max(s_min - 1)..=63.min(s_max + 1));
			let m1 = mutate(m1[i], rng) | 1;
			let m2 = mutate(m2[i], rng) | 1;
			result[i] = Self::new(m1, m2, s1, s2);
		}
		result
    }
}

/** * `crossover_rate`: the number of bits to take from one parent (so e.g. 32 means half the bits from the first
parent and half from the second) */
fn crossover(parent1: u64, parent2: u64, rng: &mut impl Rng) -> [u64; 2] {
	debug_assert!(CROSSOVER_BITS > 0 && CROSSOVER_BITS < 33);
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
	let mut bit = 0x80000000_00000000_u64;
	for _ in 0..64 {
		if rng.gen_range(0..MUTATION_ODDS) == 0_u32 {
			mask |= bit
		}
		bit >>= 1;
	}
	// println!("Mutating {} bits (expected: {:.2}, mask: {:#x})", mask.count_ones(), 64.0/m as f32, mask);
	x ^ (x & mask)
}


#[cfg(test)]
mod tests {
	use crate::utils::HasLog2;

	use super::*;

	fn calc_min_s2(s1: u32) -> u32 {
		debug_assert!(s1 <= MAX_S1);
		// To ensure x >> (s1 + (x >> s2)) has total shift <= m, need s2 >= 64 - log2(m + 1 - s1)
		64 - (MAX_TOTAL_SHIFT + 1 - s1).log2_floor() as u32
	}

	#[test]
	fn test_max_shift() {
		for s1 in 0_u32..=MAX_S1 {
			let s2 = MIN_S2[s1 as usize];
			print!("{}, {}", s1, s2);
			assert_eq!(s2, calc_min_s2(s1));
			let s = s1 + u64::MAX.wrapping_shr(s2) as u32;
			assert!(s <= MAX_TOTAL_SHIFT);
		}
	}

	#[test]
	fn test_update_w1s() {
		let mut f = MRXSM::default();
		f.update_w1s([1.0; 64]);
		for _ in 0..100 {
			f.update_w1s([1.0; 64]);
		}
		assert_eq!(1.0, f.get_loss());

		let mut rng = rand::thread_rng();
		for _ in 0..1000 {
			let mut f = MRXSM::default();
			let mut prod = 1.0;
			for _ in 0..10 {
				let w: f32 = rng.gen_range(1.0..2.0);
				f.update_w1s([w; 64]);
				prod *= w as f64;
			}
			assert!((f.get_loss() as i64 - prod.powf(0.1).round() as i64).abs() <= 1);
		}
	}
}
