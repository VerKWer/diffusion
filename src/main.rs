extern crate serde_json;
use std::{any::Any, fs::{self, File}, io::{Read, Write}, path::Path, sync::{Arc, atomic::{AtomicBool, Ordering}},
thread::{self, JoinHandle}, time::{Duration, Instant}};

use diffusion::{diffusion::DiffusionFunc, evolution::{Evolution}, globals::*, mrxsm::MRXSM};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

type F = MRXSM;  // the type of diffusion function used

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
    let results: Vec<Result<Evolution<F>, Box<dyn Any + Send>>> =
			handles.into_iter().map(|h| h.join()).collect();
	print_results(&results, &start.elapsed());
}

fn spawn_thread(thread_num: u32, interrupt: Arc<AtomicBool>, mp: &MultiProgress)
		-> JoinHandle<Evolution<F>> {
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
			},
			None => Evolution::random(&mut rng)
		};
		for i in 0..N_GENERATIONS {
			if interrupt.load(Ordering::Acquire) { break; }
			ev.next_gen(&mut rng);
			if ev.generation_counter & 31 == 0 {
				store_state(thread_num, &ev);
			}
			pb.set_position(i as u64);
			let err = ev.min_loss;
			let best = ev.get_best_func();
			pb.set_message(format!("(loss: {:.4}, age: {}, m1: {:#x}, m2: {:#x}, s1: {}, s2: {})",
				err, best.age, best.m1, best.m2, best.s1, best.s2));
		}
		pb.finish();
		// store_state(thread_num, &ev);
		ev
	})
}

fn try_load_state(thread_num: u32) -> Option<Evolution<F>> {
	let path = format!("state/{}.json", thread_num);
	let path = Path::new(path.as_str());
    let display = path.display();
	let mut file = match File::open(path) {
		Ok(f) => f,
		Err(_) => return Option::None
	};
	let mut s = String::new();
    match file.read_to_string(&mut s) {
        Err(why) => panic!("couldn't read {}: {}", display, why),
        Ok(_) => {}
    }
	let ev: Evolution<F> = match serde_json::from_str(&s) {
		Ok(e) => e,
		Err(why) => panic!("invalid state {}: {}", thread_num, why)
	};
	Option::Some(ev)
}

fn store_state(thread_num: u32, ev: &Evolution<F>) {
	fs::create_dir_all("state").unwrap();
	let mut file = File::create(format!("state/{}.json", thread_num)).unwrap();
	let serialised = serde_json::to_string(ev).unwrap();
	file.write_all(serialised.as_bytes()).unwrap();
}

fn print_results(results: &Vec<Result<Evolution<F>, Box<dyn Any + Send>>>,
		elapsed: &Duration) {
	println!("\nEvolution took {} s (generations: {}, threads: {}, time/gen: {:.2} ms)", elapsed.as_secs(),
			N_GENERATIONS, N_THREADS, elapsed.as_millis() as f64/N_GENERATIONS as f64);
	let mut best = &F::default();
	let mut oldest = &F::default();
	for (i, r) in results.iter().enumerate() {
		println!("\nResults for thread {}:", i);
		match r {
			Ok(ev) => {
				// println!("{}\n", serde_json::to_string(&ev).unwrap());
				println!("Best current: {}", ev.get_best_func());
				let thread_oldest = ev.get_longest_lived();
				println!("Oldest: {}", thread_oldest);
				let f = ev.get_best_func();
				if f.get_loss() < best.get_loss() {
					best = f;
				}
				if thread_oldest.age > oldest.age {
					oldest = thread_oldest;
				}
			},
			Err(_) => panic!("Failed printing summary!")
		}

	}
	println!("\nAbsolute lowest loss: {}", best);
	println!("Absolute oldest: {}", oldest);
	println!();
	print_by_age(results, oldest.age);
}

fn print_by_age(results: &Vec<Result<Evolution<F>, Box<dyn Any + Send>>>, oldest: u32) {
	let dummy = F::default();
	let mut best = vec![&dummy; oldest as usize + 1];
	let mut min_loss = vec![f32::MAX; oldest as usize + 1];
	for r in results.iter() {
		match r {
			Ok(ev) => {
				for f in ev.current_gen.0.iter() {
					let age = f.age as usize;
					let loss = f.get_loss();
					if loss < min_loss[age] {
						best[age] = f;
						min_loss[age] = loss;
					}
				}
			},
			Err(_) => panic!()  // can't happen
		}
	}
	println!("By age:");
	for i in 1..=oldest as usize {
		let f = best[i];
		if f.age > 0 {  // we might not have a function of that age
			println!("Age {:02}: {}", i, f);
		}
	}
}
