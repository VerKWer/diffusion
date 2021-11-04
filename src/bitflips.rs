use crate::{diffusion::Endo64, wasserstein};

#[derive(Debug, PartialEq, Eq)]
pub struct Bitflips {
	n_flips: [[u32; 65]; 64],
}

impl Bitflips {
	pub fn new(n_flips: [[u32; 65]; 64]) -> Self { Self { n_flips } }

	pub fn of(f: &impl Endo64, samples: &[u64]) -> Self {
		let mut n_flips = [[0_u32; 65]; 64];
		for &x in samples {
			let h = f.diffuse(x);
			for shift in 0..64 {
				let diff = (h ^ f.diffuse(x ^ (1 << shift))) as u64;
				n_flips[shift][diff.count_ones() as usize] += 1;
			}
		}
		Self::new(n_flips)
	}

	pub fn w1s(&self) -> [f32; 64] {
		let mut w1s = [0_f32; 64];
		for i in 0..64 {
			let counts = &self.n_flips[i];
			w1s[i] = wasserstein::of_counts(counts);
		}
		w1s
	}
}
