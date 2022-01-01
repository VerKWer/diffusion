use aligned_array::{Aligned, A32};

const LENGTH_TABLE: Aligned<A32, [u32; 256]> = init_length_table();

const fn init_length_table() -> Aligned<A32, [u32; 256]> {
	let mut result = [0_u32; 256];
	let mut byte = 0_u32;
	while byte < 256 {
		result[byte as usize] = byte.count_ones();
		byte += 1;
	}
	Aligned::<A32, _>(result)
}

pub fn pop_count(x: u8) -> u32 { LENGTH_TABLE[x as usize] }



const SET_BITS: Aligned<A32, [[u32; 8]; 256]> = init_set_bits();

const fn init_set_bits() -> Aligned<A32, [[u32; 8]; 256]> {
	let mut result = [[0_u32; 8]; 256];
	let mut byte = 0_u32;
	while byte < 256 {
		let mut i = 0_u32;
		while i < 8 {
			result[byte as usize][i as usize] = (byte >> i) & 1;
			i += 1;
		}
		byte += 1;
	}
	Aligned::<A32, _>(result)
}


/** Returns an array whose `n`<sup>th</sup> entry is 1 iff the `n`<sup>th</sup> bit of `x` (starting at the least
significant bit) is 1 and 0 otherwise. This means that (in a sense), the order of bits is reversed but more
importantly, each bit of x is expanded to a 32-bit integer.

The returned array reference is guaranteed to be 32-byte aligned. */
#[inline(always)]
pub fn get_set_bits(x: u8) -> &'static [u32; 8] { &SET_BITS[x as usize] }


#[cfg(test)]
mod tests {
    use super::*;

	#[test]
	fn test_pop_count() {
        for i in 0_u8..255 {
            assert_eq!(i.count_ones(), pop_count(i));
        }
    }

    #[test]
    fn test_set_bits() {
        for x in 0_u8..255 {
            let bs = get_set_bits(x);
            let mut y = 0;
            for (i, b) in bs.iter().enumerate() {
                y += b << i;
            }
            assert_eq!(x as u32, y);
        }
    }
}
