#![forbid(non_ascii_idents)]
#![allow(dead_code)]
#![allow(unused_imports)]
extern crate serde_json;
use std::{
	any::Any,
	fs::{self, File},
	io::{Read, Write},
	path::Path,
	sync::{
		atomic::{AtomicBool, Ordering},
		Arc,
	},
	thread::{self, JoinHandle},
	time::{Duration, Instant},
};

use diffusion::{evaluation::Evaluator, evolution::Evolution, globals::*};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

#[cfg(feature = "profile")]
fn main() {
	let mut rng = rand::thread_rng();
	let mut ev = Evolution::<F, E>::random(&mut rng);
	let start = Instant::now();
	ev.next_gen(&mut rng);
	let elapsed = start.elapsed();
	println!("Evaluation took {} ms (generation size: {}, samples: {})", elapsed.as_millis(), GENERATION_SIZE,
		N_SAMPLES);
}

#[cfg(not(feature = "profile"))]
fn main() {
	// Start evolution threads
	let mp = MultiProgress::new();
	let interrupt = Arc::new(AtomicBool::new(false));
	{
		let interrupt = Arc::clone(&interrupt);
		ctrlc::set_handler(move || interrupt.store(true, Ordering::Release)).unwrap();
	}
	let mut handles = Vec::with_capacity(N_THREADS as usize);
	let start = Instant::now();
	for n in 0..N_THREADS {
		handles.push(spawn_thread(n, Arc::clone(&interrupt), &mp));
	}
	println!();
	let _ = mp.join();
	let results: Vec<Result<Evolution<F, E>, Box<dyn Any + Send>>> = handles.into_iter().map(|h| h.join()).collect();
	print_results(&results, &start.elapsed());
}


fn spawn_thread(thread_num: u32, interrupt: Arc<AtomicBool>, mp: &MultiProgress) -> JoinHandle<Evolution<F, E>> {
	let pb = mp.add(ProgressBar::new(N_GENERATIONS as u64));
	let sty = ProgressStyle::default_bar()
		.template("[{spinner:.yellow} {elapsed_precise}/{eta_precise}] [{bar:40}] {pos:>6}/{len:6} {msg}")
		.progress_chars("#>-");
	pb.set_style(sty);
	thread::spawn(move || {
		let mut rng = rand::thread_rng();
		let mut ev = match try_load_state(thread_num) {
			Some(ev) => {
				println!("Continuing from previous state {} (gen: {})", thread_num, ev.generation_counter);
				ev
			}
			None => Evolution::random(&mut rng),
		};
		for i in 0..N_GENERATIONS {
			if interrupt.load(Ordering::Acquire) {
				break;
			}
			ev.next_gen(&mut rng);
			if ev.generation_counter & 31 == 0 {
				store_state(thread_num, &ev);
			}
			pb.set_position(i as u64);
			let best = ev.get_best();
			pb.set_message(format!("{}", best));
		}
		pb.finish();
		// store_state(thread_num, &ev);
		ev
	})
}


fn try_load_state(thread_num: u32) -> Option<Evolution<F, E>> {
	let path = format!("state/{}.json", thread_num);
	let path = Path::new(path.as_str());
	let display = path.display();
	let mut file = match File::open(path) {
		Ok(f) => f,
		Err(_) => return Option::None,
	};
	let mut s = String::new();
	if let Err(why) = file.read_to_string(&mut s) {
		panic!("couldn't read {}: {}", display, why)
	}
	let ev: Evolution<F, E> = match serde_json::from_str(&s) {
		Ok(e) => e,
		Err(why) => panic!("invalid state {}: {}", thread_num, why),
	};
	Option::Some(ev)
}


fn store_state(thread_num: u32, ev: &Evolution<F, E>) {
	fs::create_dir_all("state").unwrap();
	let mut file = File::create(format!("state/{}.json", thread_num)).unwrap();
	let serialised = serde_json::to_string(ev).unwrap();
	file.write_all(serialised.as_bytes()).unwrap();
}


#[allow(clippy::type_complexity)]
fn print_results(results: &[Result<Evolution<F, E>, Box<dyn Any + Send>>], elapsed: &Duration) {
	println!(
		"\nEvolution took {} s (generations: {}, threads: {}, time/gen: {:.2} ms)",
		elapsed.as_secs(),
		N_GENERATIONS,
		N_THREADS,
		elapsed.as_millis() as f64 / N_GENERATIONS as f64
	);
	let mut best: Option<&E> = Option::None;
	let mut oldest: Option<&E> = Option::None;
	for (i, r) in results.iter().enumerate() {
		println!("\nResults for thread {}:", i);
		match r {
			Ok(ev) => {
				// println!("{}\n", serde_json::to_string(&ev).unwrap());
				println!("Best current: {}", ev.get_best());
				let thread_oldest = ev.get_longest_lived();
				println!("Oldest: {}", thread_oldest);
				let f = ev.get_best();
				if f.get_loss() < best.map(|f| f.get_loss()).unwrap_or(f32::MAX) {
					best = Option::from(f);
				}
				if thread_oldest.get_age() > oldest.map(|f| f.get_age()).unwrap_or(0) {
					oldest = Option::from(thread_oldest);
				}
			}
			Err(_) => panic!("Failed printing summary!"),
		}
	}
	println!("\nAbsolute lowest loss: {}", best.unwrap());
	println!("Absolute oldest: {}", oldest.unwrap());
	// println!();
	// print_by_age(results, oldest.unwrap().get_age());
}


#[allow(clippy::type_complexity)]
fn print_by_age(results: &[Result<Evolution<F, E>, Box<dyn Any + Send>>], oldest: u32) {
	let mut best = vec![Option::None; oldest as usize + 1];
	let mut min_loss = vec![f32::MAX; oldest as usize + 1];
	for r in results.iter() {
		match r {
			Ok(ev) => {
				for f in ev.current_gen.members.iter() {
					let age = f.get_age() as usize;
					let loss = f.get_loss();
					if loss < min_loss[age] {
						best[age] = Option::from(f);
						min_loss[age] = loss;
					}
				}
			}
			Err(_) => panic!(), // can't happen
		}
	}
	println!("By age:");
	for (i, &f) in best.iter().enumerate() {
		if f.is_some() {
			// we might not have a function of that age
			println!("Age {:02}: {}", i, f.unwrap());
		}
	}
}
