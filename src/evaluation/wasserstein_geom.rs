use std::fmt::{self, Display, Formatter};

use rand::Rng;

use crate::{diffusion::DiffusionFunc, evaluation::bitflips::Bitflips, globals::{N_ROUNDS, N_SAMPLES}, utils::wasserstein::normalise};

use super::Evaluator;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]

/** 'Evaluator' that uses the 1-Wasserstein distance to a Binom(64, .5) as its loss function. Given a perfectly random
   function, for each input bit, the number of bit flips in the output when the input bit is flipped follows a binomial
   distribution and we measure the quality of our diffusion function by measuring the distance to that binomial
   distribution and taking the maximum over all input bits. To avoid outliers, we subdivide the samples and average the
   calculated distances for each input bit over multiple rounds.

   Over the course of the genetic algorithm the results of this loss function are averaged over the generations that
   this 'Evaluator' lives by taking the geometric mean. This is always '<=' the usual, arithmetic mean (which favours
   higher values) and so, getting unlucky in one generation will have less of an effect on the averaged score. */
pub struct WassersteinGeom<F: DiffusionFunc> {
	#[serde(with="serde_arrays")]
	w1s: [f32; 64],
	max_w1: f32,
	age: u32,
	func: F
}

impl<F: DiffusionFunc> WassersteinGeom<F> {
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
		} else if self.age == 2 {
			for (prev, &new) in self.w1s.iter_mut().zip(w1s.iter()) {
				*prev = (*prev + 1.0).log2() + (new + 1.0).log2();
				if *prev > m {
					m = *prev;
				}
			}
			m = (m / self.age as f32).exp2() - 1.0;
		} else {
			for (prev, &new) in self.w1s.iter_mut().zip(w1s.iter()) {
				*prev += (new + 1.0).log2();
				if *prev > m {
					m = *prev;
				}
			}
			m = (m / self.age as f32).exp2() - 1.0;
		}
		self.max_w1 = m;
		m
	}
}

impl<F: DiffusionFunc> Evaluator<F> for WassersteinGeom<F> {
	fn new(func: F) -> Self {
		Self { w1s: [f32::MAX; 64], max_w1: f32::MAX, age: 0, func }
	}

	fn random(rng: &mut impl Rng) -> Self {
		Self::new(F::random(rng))
	}

    fn get_age(&self) -> u32 { self.age }

    fn get_loss(&self) -> f32 { self.max_w1 }

	fn get_func(&self) -> &F { &self.func }

    fn update(&mut self, samples: &[u64; (crate::globals::N_SAMPLES) as usize]) -> f32 {
		const N_SAMPLES_PER_ROUND: u32 = N_SAMPLES/N_ROUNDS;
        let mut avg = [0_f32; 64];
		for round in 0..N_ROUNDS as usize {
			let samples = &samples[round..(round + N_SAMPLES_PER_ROUND as usize)];
			let w1s = Bitflips::of(&self.func, samples.try_into().unwrap()).w1s();
			for i in 0..64 {
				avg[i] += w1s[i];
			}
		}
		for a in avg.iter_mut() {
			*a /= N_ROUNDS as f32;
		}
		self.update_w1s(avg)
    }
}

impl<F: DiffusionFunc> Display for WassersteinGeom<F> {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		write!(f, "W1G{{")?;
		write!(f, "loss:{}, ", self.get_loss())?;
		write!(f, "age:{}", self.get_age())?;
		write!(f, "func:{}", self.get_func())?;
		write!(f, "}}")
	}
}
