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
}
