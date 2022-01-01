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
   this 'Evaluator' lives by simply taking the arithmetic mean. This tends towards higher values and so, getting unlucky
   in one generation might disproportionately affect the averaged score.  */
pub struct WassersteinArith<F: DiffusionFunc> {
	#[serde(with="serde_arrays")]
	w1s: [f32; 64],
	max_w1: f32,
	age: u32,
	func: F
}

impl<F: DiffusionFunc> WassersteinArith<F> {
	/** Updates the stored Wasserstein distances of this diffusion function and returns the updated maximum value. */
	#[inline(always)]
	fn update_w1s(&mut self, w1s: [f32; 64]) -> f32 {
		self.age += 1;
		let mut loss = 0_f32;
		if self.age == 1 {
			// this is the first SSE calculated
			self.w1s = w1s;
			for w in self.w1s {
				if w > loss {
					loss = w;
				}
			}
		} else {
			#[allow(clippy::needless_range_loop)]
			for i in 0..64 {
                let w1 = self.w1s[i] + w1s[i];
				self.w1s[i] = w1;
				if w1 > loss {
					loss = w1;
				}
			}
			loss /= self.age as f32;
		}
		self.max_w1 = loss;
		loss
	}
}

impl<F: DiffusionFunc> Evaluator<F> for WassersteinArith<F> {
	fn new(func: F) -> Self {
		Self { w1s: [f32::MAX; 64], max_w1: f32::MAX, age: 0, func }
	}

	fn random(rng: &mut impl Rng) -> Self {
		Self::new(F::random(rng))
	}

    fn get_age(&self) -> u32 { self.age }

    fn get_loss(&self) -> f32 { self.max_w1 }

	fn get_func(&self) -> &F { &self.func }

    fn update(&mut self, samples: &[u64; N_SAMPLES as usize]) -> f32 {
		const N_SAMPLES_PER_ROUND: u32 = N_SAMPLES/N_ROUNDS;
        let mut avg = [0_f32; 64];
		for round in 0..N_ROUNDS {
			let l = (round * N_SAMPLES_PER_ROUND) as usize;
			let samples = &samples[l..(l + N_SAMPLES_PER_ROUND as usize)];
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

impl<F: DiffusionFunc> Display for WassersteinArith<F> {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		write!(f, "W1A{{")?;
		write!(f, "loss:{}, ", self.get_loss())?;
		write!(f, "age:{}, ", self.get_age())?;
		write!(f, "func:{}", self.get_func())?;
		write!(f, "}}")
	}
}


#[cfg(test)]
mod tests {
	use super::*;
    use crate::{diffusion::mrxsm::MRXSM};

	#[test]
	fn test_update_w1s() {
		let f = MRXSM::default();
		let mut e = WassersteinArith::new(f);
		e.update_w1s([1.0; 64]);
		for _ in 0..100 {
			e.update_w1s([1.0; 64]);
		}
		assert_eq!(1.0, e.get_loss());

		let mut rng = rand::thread_rng();
		for _ in 0..1000 {
			let mut prod = 1.0;
			for _ in 0..10 {
				let w: f32 = rng.gen_range(1.0..2.0);
				e.update_w1s([w; 64]);
				prod *= w as f64;
			}
			assert!((e.get_loss() as i64 - prod.powf(0.1).round() as i64).abs() <= 1);
		}
	}
}
