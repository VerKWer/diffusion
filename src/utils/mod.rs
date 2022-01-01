#![allow(clippy::missing_safety_doc)]
use std::{arch::x86_64::*, mem::{self}};

use aligned_array::{A32, Aligned};
use num_integer::{div_rem};
use rand::Rng;

pub mod bitset;
pub mod wasserstein;

/** For some reason _mm256_load intrinsics are painfully slow and using transmute is significantly faster.
 * However, you need to make sure yourself that the pointer is 32-byte aligned. */
#[inline(always)]
pub unsafe fn read_m256i<T>(ptr: *const T) -> __m256i {
    let ptr = ptr as *const __m256i;
    ptr.read()
}

#[inline(always)]
pub unsafe fn write_m256i<T>(v: __m256i, ptr: *mut T) {
    let ptr = ptr as *mut __m256i;
    ptr.write(v);
}

#[inline(always)]
pub fn m128i_from_u32(x: u32) -> __m128i {
    unsafe { _mm_set_epi32(0, 0, 0, x as i32) }
}

#[inline(always)]
pub fn m256i_from_u32(x: u32) -> __m256i {
    unsafe { _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, x as i32) }
}

#[inline(always)]
pub fn m256i_from_u64(x: u64) -> __m256i {
    unsafe { _mm256_set_epi64x(0, 0, 0, x as i64) }
}

#[inline(always)]
pub fn m256i_from_u32x8(v: [u32; 8]) -> __m256i {
    unsafe { mem::transmute::<[u32; 8], __m256i>(v) }
}

#[inline(always)]
pub fn m256i_from_u64x4(v: [u64; 4]) -> __m256i {
    unsafe { mem::transmute::<[u64; 4], __m256i>(v) }
}

#[inline(always)]
pub fn m256i_to_i32x8(v: __m256i) -> [i32; 8] {
    unsafe { mem::transmute::<__m256i, [i32; 8]>(v) }
}

#[inline(always)]
pub fn m256i_to_u32x8(v: __m256i) -> [u32; 8] {
    unsafe { mem::transmute::<__m256i, [u32; 8]>(v) }
}

#[inline(always)]
pub fn m256i_to_u64x4(v: __m256i) -> [u64; 4] {
    unsafe { mem::transmute::<__m256i, [u64; 4]>(v) }
}

#[inline(always)]
pub fn mul_m256i64(a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let mixed = _mm256_shuffle_epi32::<0b10_11_00_01>(b);  // [b_il, b_ih]_i
        let mixed = _mm256_mullo_epi32(a, mixed);  // [a_ih * b_il, a_il * b_ih]_i
        let shifted = _mm256_slli_epi64::<32>(mixed);  // [a_il * b_ih, 0]_i
        let mixed = _mm256_add_epi32(mixed, shifted);  // [a_ih * b_il + a_il * b_ih, a_il * b_ih]_i
        // let mixed = _mm256_blend_epi32::<0b01010101>(mixed, shifted);  // [a_ih * b_il + a_il * b_ih, 0]_i
        // Using _mm256_and has better latency and throughput than blend
        #[allow(overflowing_literals)]
        let mixed = _mm256_and_si256(mixed, _mm256_set1_epi64x(0xffffffff00000000));

        let prod = _mm256_mul_epu32(a, b);  // product of lower 32 bits: [a_il * b_il]_i  (64 bit ints!)
        _mm256_add_epi64(mixed, prod)
    }
}

/** Takes the horizontal sum of eight 32 bit integers in a vector. */
#[inline(always)]
pub fn hsum_m256i32(vals: __m256i) -> i32 {
    unsafe {
        let high = _mm256_extracti128_si256::<1>(vals);  // extract top 128 bits [x7, x6, x5, x4]
        let acc = _mm256_castsi256_si128(vals);  // bottom 128 bits (generates no instructions) [x3, x2, x1, x0]
        let acc = _mm_add_epi32(acc, high);  // [x3+x7, x2+x6, x1+x5, x0+x4] =: [y3, y2, y1, y0]

        /* We could also use right shift (vpsrldq), which would be faster on older Intel Haswell. However, for Zen2, the
        docs suggest that latency and throughput for shuffle are better and it is available on more ports. */
        let high = _mm_shuffle_epi32::<0b00_00_11_10>(acc);  // [y0, y0, y3, y2]
        let acc = _mm_add_epi32(acc, high);  // [y3+y0, y2+y0, y1+y3, y0+y2] =: [_, _, z1, z0]
        let high = _mm_shuffle_epi32::<0b00_00_01_01>(acc);  // [z0, z0, z1, z1]
        let acc = _mm_add_epi32(acc, high);  // [_, _, _, z0+z1]
        _mm_extract_epi32::<0>(acc)  // z0 + z1 = y0 + y2 + y1 + y3 = x0 + x4 + x2 + x6 + x1 + x5 + x3 + x7
    }
}

/** Takes the horizontal xor of four 64 bit integers in a vector. */
#[inline(always)]
pub fn hxor_m256i64(vals: __m256i) -> i64 {
    unsafe {
        let high = _mm256_extracti128_si256::<1>(vals);  // extract top 128 bits [x3, x2]
        let acc = _mm256_castsi256_si128(vals);  // bottom 128 bits (generates no instructions) [x1, x0]
        let acc = _mm_xor_epi64(acc, high);  // [x1^x3, x0^x2] =: [y1, y0]

        /* We could also use right shift (vpsrldq), which would be faster on older Intel Haswell. However, for Zen2, the
        docs suggest that latency and throughput for shuffle are better and it is available on more ports. */
        let flipped = _mm_shuffle_epi32::<0b01_00_11_10>(acc);  // [y0, y1]
        let acc = _mm_xor_epi64(acc, flipped);  // [y1^y0, y0^y1] = [x1^x3^x0^x2, x0^x2^x1^x3]
        _mm_extract_epi64::<0>(acc)
    }
}

#[inline(always)]
pub fn xor_many<const LENGTH: usize>(vals: &Aligned<A32, [u64; LENGTH]>, to_xor: u64) -> Aligned<A32, [u64; LENGTH]> {
    debug_assert!(LENGTH & 3 == 0);
    let result: Aligned<A32, _> = Aligned([0_u64; LENGTH]);
    let mut vals_ptr = vals.as_ptr() as *const i64;
    let mut result_ptr = result.as_ptr() as *mut i64;
    unsafe {
        let xs = _mm256_set1_epi64x(to_xor as i64);
        for _ in 0..(LENGTH >> 2) {
            let v = read_m256i(vals_ptr);
            let v = _mm256_xor_epi64(v, xs);
            write_m256i(v, result_ptr);
            vals_ptr = vals_ptr.offset(4);
            result_ptr = result_ptr.offset(4);
        }
        result
    }
}

pub trait HasLog2 {
    fn log2_floor(self) -> i32;
    fn log2_ceil(self) -> i32;
}

impl HasLog2 for u32 {
    #[inline(always)]
    fn log2_floor(self) -> i32 {
        31 - self.leading_zeros() as i32
    }
    #[inline(always)]
    fn log2_ceil(self) -> i32 {
        self.log2_floor() + ((self.count_ones() > 1) as i32)
    }
}

impl HasLog2 for u64 {
    #[inline(always)]
    fn log2_floor(self) -> i32 {
        63 - self.leading_zeros() as i32
    }
    #[inline(always)]
    fn log2_ceil(self) -> i32 {
        self.log2_floor() + ((self.count_ones() > 1) as i32)
    }
}


/** Returns a (pseudo-)random 32-bit integer sampled from a (shifted) geometric distribution with probability 1/n.

   _Note:_ The smallest possible result is 1. */
#[inline]
pub fn random_geom_u32(n: u32, rng: &mut impl Rng) -> u32 {
    let mut result = 1_u32;
    let mut x: u32;
    loop {
        x = rng.gen();
        if x != 0 { break; }
        result += (32.0 / (x as f64).log2()).ceil() as u32;  // extremely unlikely
    }
    loop {
        let (y, r) = div_rem(x, n);
        if r != 0 { break; }
        result += 1;
        x = y;
    }
    result
}



#[cfg(test)]
mod tests {
    use std::{collections::{HashMap, hash_map::Entry}};

    use aligned_array::{A32, Aligned};
    use rand::Rng;
    use super::*;

    #[test]
    fn test_read_write() {
        let a: Aligned<A32,_> = Aligned([0_i64, 1, 2, 3]);
        let v = unsafe{ read_m256i(a.as_ptr()) };
        let mut b: Aligned<A32, _> = Aligned([0_i64; 4]);
        // unsafe { _mm256_store_si256(b.as_mut_ptr() as *mut __m256i, v)}
        unsafe { write_m256i(v, b.as_mut_ptr()) };
        assert_eq!(a, b);
    }

    #[test]
    fn test_blend() {
        let v = m256i_from_u32x8([8, 7, 6, 5, 4, 3, 2, 1]);
        let w = m256i_from_u32x8([0, 0, 0, 0, 0, 0, 0, 0]);
        let v = unsafe { _mm256_blend_epi32::<0b00001111>(v, w) };
        let v = m256i_to_u32x8(v);
        dbg!(v);
    }

    #[test]
    fn test_log2_u32() {
        assert_eq!(31, u32::MAX.log2_floor());
        assert_eq!(32, u32::MAX.log2_ceil());
        assert_eq!(-1, 0u32.log2_floor());
        assert_eq!(-1, 0_u32.log2_ceil());
        assert_eq!(0, 1_u32.log2_floor());
        assert_eq!(0, 1_u32.log2_ceil());
        let mut rng = rand::thread_rng();
        for s in 1..32 {
            let msb = 1_u32 << s;
            assert_eq!(s, msb.log2_floor());
            assert_eq!(s, msb.log2_ceil());
            for _ in 0..10 {
                let x = msb + rng.gen_range(1..msb);
                assert_eq!(s, x.log2_floor());
                assert_eq!(s + 1, x.log2_ceil());
            }
        }
    }

    #[test]
    fn test_log2_u64() {
        assert_eq!(63, u64::MAX.log2_floor());
        assert_eq!(64, u64::MAX.log2_ceil());
        assert_eq!(-1, 0_u64.log2_floor());
        assert_eq!(-1, 0_u64.log2_ceil());
        assert_eq!(0, 1_u64.log2_floor());
        assert_eq!(0, 1_u64.log2_ceil());
        let mut rng = rand::thread_rng();
        for s in 1..64 {
            let msb = 1_u64 << s;
            assert_eq!(s, msb.log2_floor());
            assert_eq!(s, msb.log2_ceil());
            for _ in 0..10 {
                let x = msb + rng.gen_range(1..msb);
                assert_eq!(s, x.log2_floor());
                assert_eq!(s + 1, x.log2_ceil());
            }
        }
    }

    #[test]
    fn test_hsum() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let xs: [i32; 8] = rng.gen();
            let v = unsafe { _mm256_loadu_epi32(xs.as_ptr()) };
            let mut sum = 0_i32;
            for x in xs {
                sum = sum.overflowing_add(x).0;
            }
            assert_eq!(sum, hsum_m256i32(v));
        }
    }

    #[test]
    fn test_hxor() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let xs: [i64; 4] = rng.gen();
            let v = unsafe { _mm256_loadu_epi64(xs.as_ptr()) };
            let mut xor = 0_i64;
            for x in xs {
                xor ^= x;
            }
            assert_eq!(xor, hxor_m256i64(v));
        }
    }

    #[test]
    fn test_shift() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let xs: [u64; 4] = rng.gen();
            let s = rng.gen_range(0..63);
            let v = m256i_from_u64x4(xs);
            let s2 = m128i_from_u32(s);
            let v = unsafe { _mm256_srl_epi64(v, s2) };
            let ys = m256i_to_u64x4(v);

            for (x, y) in xs.iter().zip(ys.iter()) {
                assert_eq!(x >> s, *y);
            }
        }
    }

    #[test]
    fn test_mul() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let xs: [u64; 4] = rng.gen();
            let ys: [u64; 4] = rng.gen();
            let mut zs = [0_u64; 4];
            for i in 0..4 { zs[i] = xs[i].overflowing_mul(ys[i]).0; }
            let zs2 = m256i_to_u64x4(mul_m256i64(m256i_from_u64x4(xs), m256i_from_u64x4(ys)));
            assert_eq!(zs, zs2);
        }

    }

    #[test]
    fn test_xor_many() {
        let mut rng = rand::thread_rng();
        let x: u64 = rng.gen();
        for _ in 0..1000 {
            let mut vals: Aligned<A32, _> = Aligned([0_u64; 128]);
            for i in 0..128 { vals[i] = rng.gen(); }
            let xored = xor_many(&vals, x);
            for i in 0..128 {
                assert_eq!(vals[i] ^ x, xored[i]);
            }
        }
    }

    #[test]
    fn test_random_geom() {
        let mut rng = rand::thread_rng();
        let mut samples = HashMap::<u32, u32>::new();
        for _ in 0..1000 {
            let x = random_geom_u32(3, &mut rng);
            match samples.entry(x) {
                Entry::Occupied(mut e) => *e.get_mut() += 1,
                Entry::Vacant(e) => { e.insert(1); }
            }
        }
        let mut l: Vec<(u32, u32)> = vec![];
        for (k, v) in samples.iter() {
            l.push((*k, *v));
        }
        l.sort_unstable_by(|p, q| { p.0.cmp(&q.0) });
        for (k, v) in l.iter() {
            println!("{}: {}", k, v);
        }
    }
}
