use std::mem::MaybeUninit;

use crate::{diffusion::DiffusionFunc, generation::Generation, globals::{ELITISM, GENERATION_SIZE, N_ROUNDS, N_SAMPLES, SAMPLE_LIFETIME_MASK}};
use rand::Rng;


#[derive(Debug, Serialize, Deserialize)]
pub struct Evolution<F> {
	#[serde(skip, default="random_samples")]
	samples: [u64; (N_SAMPLES * N_ROUNDS) as usize],
	pub generation_counter: u32,
	pub current_gen: Generation<F>,
	pub min_loss: f32,
	pub best_idx: usize,
}

fn random_samples() -> [u64; (N_SAMPLES * N_ROUNDS) as usize] {
	let mut rng = rand::thread_rng();
	const LEN: usize = (N_SAMPLES * N_ROUNDS) as usize;
	let mut samples = [0_u64; LEN];
	for i in 0..LEN {
		samples[i] = rng.gen();
	}
	samples
}


impl<F: DiffusionFunc + Clone> Evolution<F> {
	pub fn new(samples: [u64; (N_SAMPLES * N_ROUNDS) as usize], starting_gen: Generation<F>) -> Self {
		Self { samples, generation_counter: 0, current_gen: starting_gen, min_loss: f32::MAX, best_idx: 0 }
	}

	pub fn random(rng: &mut impl Rng) -> Self {
		let samples = random_samples();
		let current_gen = Generation::random(rng);
		Self { samples, generation_counter: 0, current_gen, min_loss: f32::MAX, best_idx: 0 }
	}

	pub fn get_best_func(&self) -> &F { &self.current_gen.0[self.best_idx] }

	pub fn get_longest_lived(&self) -> &F {
		let mut max_age = 0_u32;
		let mut idx = 0_usize;
		for (i, f) in self.current_gen.0.iter().enumerate() {
			if f.get_age() > max_age {
				max_age = f.get_age();
				idx = i;
			}
		}
		&self.current_gen.0[idx]
	}

	pub fn next_gen(&mut self, rng: &mut impl Rng) {
		self.eval_current_gen();
		let mut next_gen: [MaybeUninit<F>; GENERATION_SIZE as usize] =
			unsafe { MaybeUninit::uninit().assume_init() };

		self.current_gen.0.sort_unstable_by(|f, g| f.get_loss().partial_cmp(&g.get_loss()).unwrap());
		for i in 0..ELITISM {
			// println!("Adding elite {}", &self.current_gen.members[i as usize]);
			next_gen[i as usize] = MaybeUninit::new(self.current_gen.0[i as usize].clone())
		}

		let mut i = ELITISM as usize;
		'outer: loop {
			let parent1 = self.tournament(rng);
			let parent2 = self.tournament(rng);
			for child in parent1.crossover(parent2, rng) {
				// println!{"Adding child {}", child}
				next_gen[i] = MaybeUninit::new(child);
				i += 1;
				if i == GENERATION_SIZE as usize {
					break 'outer;
				}
			}
		}

		let ptr = &mut next_gen as *mut _ as *mut Generation<F>;
		let cast = unsafe { ptr.read() };
		core::mem::forget(next_gen);
		self.current_gen = cast;

		// Update generation counter and generate new samples for next round if necessary.
		self.generation_counter += 1;
		if self.generation_counter & SAMPLE_LIFETIME_MASK == 0 {
			self.new_samples(rng);
		}
	}

	fn eval_current_gen(&mut self) {
		let mut best_idx = 0;
		let mut min_error = f32::MAX;
		// If we just switched samples, evaluate everyting. Otherwise, only children.
		let i0 = if self.generation_counter & SAMPLE_LIFETIME_MASK == 0 { 0_usize } else { ELITISM as usize };
		for i in i0..GENERATION_SIZE as usize {
			// println!("Evaluating function {}", i);
			let f = &mut self.current_gen.0[i as usize];
			let error = f.update(&self.samples);
			if error < min_error {
				min_error = error;
				best_idx = i;
			}
		}
		// Store results
		if i0 == 0 || min_error < self.min_loss {
			self.min_loss = min_error;
			self.best_idx = best_idx;
		}
	}

	fn tournament(&self, rng: &mut impl Rng) -> &F {
		let f1 = &self.current_gen.0[rng.gen_range(0..GENERATION_SIZE as usize)];
		let f2 = &self.current_gen.0[rng.gen_range(0..GENERATION_SIZE as usize)];
		if f1.get_loss() <= f2.get_loss() {
			f1
		} else {
			f2
		}
	}

	fn new_samples(&mut self, rng: &mut impl Rng) {
		// let mut next_idx = u32::MAX;
		// loop {
		//     next_idx = next_idx.wrapping_add(1).wrapping_add(rng.gen::<u32>().trailing_zeros());
		//     if next_idx as usize >= N_SAMPLES { break; }
		//     self.samples[next_idx as usize] = rng.gen();
		// }
		for i in 0..(N_SAMPLES * N_ROUNDS) as usize {
			self.samples[i] = rng.gen();
		}
	}
}


#[cfg(test)]
mod tests {

	use crate::mrxsm::MRXSM;

	use super::*;

	#[test]
	fn test_eval() {
		let mut rng = rand::thread_rng();
		let mut ev = Evolution::<MRXSM>::random(&mut rng);
		ev.eval_current_gen();
		println!("min_error: {}", ev.min_loss);
		// let min_sse = AvalancheDiagram::of(&ev.best_func, &ev.samples).sse();
		// assert_eq!(min_sse, ev.min_error);
	}
}
