use std::mem::MaybeUninit;

use crate::{
	diffusion::DiffusionFunc,
	evaluation::Evaluator,
	globals::{ELITISM, GENERATION_SIZE, N_SAMPLES, SAMPLE_LIFETIME, TOURNAMENT_SIZE},
};
use rand::Rng;

use self::generation::Generation;

mod generation;


const SAMPLE_LIFETIME_MASK: u32 = SAMPLE_LIFETIME - 1;


#[derive(Debug, Serialize, Deserialize)]
pub struct Evolution<F, E> {
	#[serde(skip, default = "random_samples")]
	samples: [u64; N_SAMPLES as usize],
	pub generation_counter: u32,
	pub current_gen: Generation<F, E>,
}

fn random_samples() -> [u64; N_SAMPLES as usize] {
	let mut rng = rand::thread_rng();
	const LEN: usize = N_SAMPLES as usize;
	let mut samples = [0_u64; LEN];
	for sample in samples.iter_mut() {
		*sample = rng.gen();
	}
	samples
}


impl<F: DiffusionFunc, E: Evaluator<F>> Evolution<F, E> {
	pub fn new(samples: [u64; N_SAMPLES as usize], starting_gen: Generation<F, E>) -> Self {
		Self { samples, generation_counter: 0, current_gen: starting_gen }
	}

	pub fn random(rng: &mut impl Rng) -> Self {
		let samples = random_samples();
		let current_gen = Generation::random(rng);
		Self { samples, generation_counter: 0, current_gen }
	}

	pub fn get_best(&self) -> &E {
		const LAST: usize = (GENERATION_SIZE - 1) as usize;
		// After next_gen returns, the best function will be at the end of the array.
		&self.current_gen.members[LAST]
	}

	pub fn get_longest_lived(&self) -> &E {
		let mut max_age = 0_u32;
		let mut idx = 0_usize;
		for (i, f) in self.current_gen.members.iter().enumerate() {
			if f.get_age() > max_age {
				max_age = f.get_age();
				idx = i;
			}
		}
		&self.current_gen.members[idx]
	}

	pub fn next_gen(&mut self, rng: &mut impl Rng) {
		self.eval_current_gen();
		const N_CHILDREN: usize = (GENERATION_SIZE - ELITISM) as usize;
		let mut next_gen: [MaybeUninit<E>; N_CHILDREN] = unsafe { MaybeUninit::uninit().assume_init() };

		// Generate children
		for i in (0..N_CHILDREN).step_by(2) {
			let parent1 = self.tournament(rng).get_func();
			let parent2 = self.tournament(rng).get_func();
			let [child1, child2] = parent1.crossover(parent2, rng);
			next_gen[i] = MaybeUninit::new(E::new(child1));
			next_gen[i+1] = MaybeUninit::new(E::new(child2));
		}
		let ptr = &next_gen as *const _ as *const [E; N_CHILDREN];
		let cast = unsafe { ptr.read() };
		core::mem::forget(next_gen);

		// Write back children, keeping elite at the end
		for (i, f) in cast.into_iter().take(N_CHILDREN).enumerate() {
			self.current_gen.members[i] = f;
		}

		// Update generation counter and generate new samples for next round if necessary.
		self.generation_counter += 1;
		if self.generation_counter & SAMPLE_LIFETIME_MASK == 0 {
			self.new_samples(rng);
		}
	}

	fn eval_current_gen(&mut self) {
		// If we just switched samples, evaluate everyting. Otherwise, only children.
		let i0 = if self.generation_counter & SAMPLE_LIFETIME_MASK == 0 { 0_usize } else { ELITISM as usize };
		for i in i0..GENERATION_SIZE as usize {
			// println!("Evaluating function {}", i);
			let ev = &mut self.current_gen.members[i as usize];
			ev.update(&self.samples);
		}
		self.current_gen.members.sort_unstable_by(|f, g| g.get_loss().partial_cmp(&f.get_loss()).unwrap());
	}

	/** Performs a deterministic tournament (i.e. fittest competitor always wins) of size `TOURNAMENT_SIZE`. */
	fn tournament(&self, rng: &mut impl Rng) -> &E {
		let mut best: &E = &self.current_gen.members[rng.gen_range(0..GENERATION_SIZE as usize)];
		for _ in 1..TOURNAMENT_SIZE {
			let cand = &self.current_gen.members[rng.gen_range(0..GENERATION_SIZE as usize)];
			if cand.get_loss() < best.get_loss() {
				best = cand;
			}
		}
		best
	}

	fn new_samples(&mut self, rng: &mut impl Rng) {
		// let mut next_idx = u32::MAX;
		// loop {
		//     next_idx = next_idx.wrapping_add(1).wrapping_add(rng.gen::<u32>().trailing_zeros());
		//     if next_idx as usize >= N_SAMPLES { break; }
		//     self.samples[next_idx as usize] = rng.gen();
		// }
		for i in 0..N_SAMPLES as usize {
			self.samples[i] = rng.gen();
		}
	}
}


#[cfg(test)]
mod tests {

	use super::*;
	use crate::{diffusion::mrxsm::MRXSM, evaluation::wasserstein_arith::WassersteinArith};

	#[test]
	fn test_eval() {
		let f = MRXSM::new(0xb520c891288cb35, 0xb018200835e0008d, 21, 59);
		let mut rng = rand::thread_rng();
		let current_gen = Generation::random(&mut rng);
		let mut members = current_gen.members;
		members[0] = WassersteinArith::new(f);
		let current_gen = Generation::new(members);
		let mut ev = Evolution::new(random_samples(), current_gen);
		ev.eval_current_gen();
		println!("{}", ev.current_gen);
		// let min_sse = AvalancheDiagram::of(&ev.best_func, &ev.samples).sse();
		// assert_eq!(min_sse, ev.min_error);
	}
}
