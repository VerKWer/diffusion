use crate::{globals::N_SAMPLES};

// pub const BINOM64_PMF: [f32; 65] = [5.421010862427522e-20, 3.469446951953597e-18, 1.0928757898653858e-16,
//     2.258609965721796e-15, 3.4443801977257506e-14, 4.1332562372709e-13, 4.064368633316391e-12, 3.367619724747855e-11,
//     2.39942905388286e-10, 1.4929780779715472e-09, 8.211379428843538e-09, 4.031040810523176e-08,
// 1.7803763579810702e-07,     7.121505431924307e-07, 2.5942626930581445e-06, 8.647542310193826e-06,
// 2.6483098324968617e-05, 7.477580703520505e-05,     0.00019524794059192426, 0.00047270554038044847,
// 0.0010635874658560115, 0.002228468976079259, 0.004355643907791282,     0.007953784527271046, 0.013587715234088001,
// 0.02174034437454079, 0.03261051656181125, 0.04589628256847506,     0.060648659108342017, 0.07528799061725214,
// 0.08783598905346111, 0.09633624605863457, 0.09934675374796692,     0.09633624605863458, 0.08783598905346109,
// 0.07528799061725215, 0.060648659108342017, 0.045896282568475076,     0.03261051656181116, 0.021740344374540848,
// 0.013587715234088004, 0.007953784527271046, 0.0043556439077912824,     0.0022284689760792595, 0.0010635874658560115,
// 0.00047270554038044847, 0.0001952479405919242,     7.477580703520506e-05, 2.6483098324968624e-05,
// 8.647542310193826e-06, 2.5942626930581454e-06, 7.121505431924307e-07,     1.78037635798107e-07,
// 4.0310408105231766e-08, 8.211379428843538e-09, 1.4929780779715472e-09, 2.39942905388286e-10,
//     3.367619724747855e-11, 4.0643686333163915e-12, 4.1332562372708997e-13, 3.4443801977257506e-14,
//     2.258609965721796e-15, 1.092875789865386e-16, 3.469446951953596e-18, 5.421010862427522e-20];
const BINOM64_PMF: [f32; 65] = init_binom64_pmf();

const fn init_binom64_pmf() -> [f32; 65] {
	let mut pmf = [0_f32; 65];
	let mut binom_coeffs = [[0_u64; 65]; 65];
	let mut n = 0;
	while n <= 64 {
		binom_coeffs[n][0] = 1;
		let mut k = 1;
		while k <= n {
			binom_coeffs[n][k] = binom_coeffs[n - 1][(k - 1)] + binom_coeffs[n - 1][k];
			k += 1;
		}
		n += 1;
	}
	let mut k = 0;
	while k <= 64 {
		let p = binom_coeffs[64][k] as f64 / 18446744073709551616.0;
		pmf[k] = p as f32;
		k += 1;
	}

	pmf
}

const BINOM64_CDF_SCALED: [f32; 65] = init_binom64_cdf_scaled();

const fn init_binom64_cdf_scaled() -> [f32; 65] {
	let mut pmf = [0_f32; 65];
	let mut binom_coeffs = [[0_u64; 65]; 65];
	let mut n = 0;
	while n <= 64 {
		binom_coeffs[n][0] = 1;
		let mut k = 1;
		while k <= n {
			binom_coeffs[n][k] = binom_coeffs[n - 1][(k - 1)] + binom_coeffs[n - 1][k];
			k += 1;
		}
		n += 1;
	}
	let mut p = 0_f64;
	let mut k = 0;
	while k <= 64 {
		p += binom_coeffs[64][k] as f64 / 18446744073709551616.0 * N_SAMPLES as f64;
		pmf[k] = p as f32;
		k += 1;
	}

	pmf
}

/** Calculates the 1-Wasserstein distance of a finite pmf on 0..=64 to Bin(64, 0.5). */
#[inline(always)]
pub fn of_distr(p: &[f32; 65]) -> f32 {
	let mut d = 0_f32;
	let mut t = 0_f32;
	for i in 0..65 {
		t = p[i] + t - BINOM64_PMF[i];
		d += t.abs();
	}
	d //* (N_SAMPLES as f32).sqrt()
}

/** Calculates the normalised 1-Wasserstein distance from the raw case counts. This might be more efficient than scaling
    `p` itself, which would involve one floating point division per entry. Instead, we can avoid all this and only
    perform a single division at the end.

_NOTE:_ The sum of the `p` vector must equal `N_SAMPLES`. */
#[inline(always)]
pub fn of_counts(p: &[u32; 65]) -> f32 {
	let mut d = 0_f32;
	let mut p_sum = 0_u32;
	for i in 0..65 {
		p_sum += p[i];
		d += (p_sum as f32 - BINOM64_CDF_SCALED[i]).abs();
	}
	d / (N_SAMPLES as f32).sqrt()
}


#[cfg(test)]
mod tests {
	use rand::Rng;

	use super::*;

	#[test]
	fn print_binom64_pmf() {
		dbg!(BINOM64_PMF);
	}

	#[test]
	fn test_binom_coeffs() {
		let mut binom_coeffs = [[0_u64; 65]; 65];
		for n in 0..=64 {
			binom_coeffs[n][0] = 1;
			for k in 1..=n {
				binom_coeffs[n][k] = binom_coeffs[n - 1][(k - 1)] + binom_coeffs[n - 1][k];
			}
		}
		assert_eq!(1, binom_coeffs[2][0]);
		assert_eq!(2, binom_coeffs[2][1]);
		assert_eq!(1, binom_coeffs[2][2]);
		assert_eq!(1832624140942590534, binom_coeffs[64][32]);
	}

	#[test]
	fn test_wasserstein() {
		assert_eq!(0.0, of_distr(&BINOM64_PMF));
		let mut rng = rand::thread_rng();
		let mut counts = [0_u32; 65];
		for _ in 0..10_000 {
			let c = rng.gen::<u64>().count_ones();
			counts[c as usize] += 1;
		}
		let sum: u32 = counts.iter().sum();
		// Should tend towards Bin(64, 0.5) distribution.
		let p = counts.map(|c| c as f32 / sum as f32);
		let w = of_distr(&p);
		println!("{}", w);
		assert!(w < 0.2);
	}

	#[test]
	fn test_convergence() {
		let mut rng = rand::thread_rng();
		let n_samples = 1000_u32;
		let n_rounds = 1000_u32;
		let mut avg = 0_f64;
		for _ in 0..n_rounds {
			let mut counts = [0_u32; 65];
			for _ in 0..n_samples {
				let c = rng.gen::<u64>().count_ones();
				counts[c as usize] += 1;
			}
			let sum: u32 = counts.iter().sum();
			// Should tend towards Bin(64, 0.5) distribution.
			let p = counts.map(|c| c as f32 / sum as f32);
			let w = of_distr(&p);
			avg += w as f64;
		}
		println!("{}", avg / (n_rounds as f64) * (n_samples as f64).sqrt());
	}

	#[test]
	fn test_wasserstein_of_counts() {
		let mut rng = rand::thread_rng();
		for _ in 0..100 {
			let mut counts = [0_u32; 65];
			for _ in 0..N_SAMPLES {
				let i = rng.gen_range(0..65);
				counts[i] += 1;
			}
			let p = counts.map(|c| c as f32 / N_SAMPLES as f32);
			let w1 = of_distr(&p) * (N_SAMPLES as f32).sqrt();
			let w2 = of_counts(&counts);
			assert!((w1 - w2).abs() <= 10e-5);
		}
	}
}
