use aligned_array::{A32, Aligned};
use criterion::{criterion_group, criterion_main, Criterion};
use diffusion::{diffusion::{DiffusionFunc, Endo64}, mrxsm::{MRXSM}, utils};
use rand::Rng;
use rand_distr::{Distribution, Geometric};


fn diffuse_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let mut xs: Aligned<A32, _> = Aligned([0_u64; 4096]);
    for i in 0..xs.len() { xs[i] = rng.gen(); }
    let f = MRXSM::random(&mut rng);

    c.bench_function("diffuse_sequential", |b| {
        b.iter(|| {
            let mut ys = [0_u64; 4096];
            for i in 0..4096 {
                ys[i] = f.diffuse(xs[i]);
            }
            ys
        })
    });
}

fn xor_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let mut xs: Aligned<A32, _> = Aligned([0_u64; 1024]);
    for i in 0..xs.len() { xs[i] = rng.gen(); }
    let x: u64 = rng.gen();

    c.bench_function("xor_sequential", |b| {
        b.iter(|| {
            let mut ys = [0_u64; 1024];
            for i in 0..1024 {
                ys[i] = xs[i] ^ x;
            }
            ys
        })
    });

    c.bench_function("xor_simd", |b| {
        b.iter(|| {
            utils::xor_many(&mut xs, x)
        })
    });
}

fn geom_distr_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let geo = Geometric::new(0.33333).unwrap();

    c.bench_function("rand_geo", |b| {
        b.iter(|| {
            let mut ys = [0_u32; 1024];
            for i in 0..1024 {
                ys[i] = geo.sample(&mut rng) as u32;
            }
            ys
        })
    });

    c.bench_function("our_geo", |b| {
        b.iter(|| {
            let mut ys = [0_u32; 1024];
            for i in 0..1024 {
                ys[i] = utils::random_geom_u32(3, &mut rng);
            }
            ys
        })
    });

    c.bench_function("our_geo_half", |b| {
        b.iter(|| {
            let mut ys = [0_u32; 1024];
            for i in 0..1024 {
                ys[i] = rng.gen::<u32>().trailing_zeros();
            }
            ys
        })
    });
}

fn sample_exchange_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let mut samples = [0_u64; 1000];
    let geo = Geometric::new(0.5).unwrap();

    c.bench_function("naive_exchange", |b| {
        b.iter(|| {
            for i in 0..samples.len() {
                let change: bool = rng.gen();
                samples[i] = if change { rng.gen() } else { samples[i] };
            }
        })
    });

    c.bench_function("rng_geo_exchange", |b| {
        b.iter(|| {
            let mut next_idx = u32::MAX;
            loop {
                next_idx = next_idx.wrapping_add(1).wrapping_add(geo.sample(&mut rng) as u32);
                if next_idx as usize >= samples.len() { break; }
                samples[next_idx as usize] = rng.gen();
            }
        })
    });

    c.bench_function("trailing_zeros_exchange", |b| {
        b.iter(|| {
            let mut next_idx = u32::MAX;
            loop {
                next_idx = next_idx.wrapping_add(1).wrapping_add(rng.gen::<u32>().trailing_zeros());
                if next_idx as usize >= samples.len() { break; }
                samples[next_idx as usize] = rng.gen();
            }
        })
    });
}

criterion_group!(diffuse, diffuse_benchmark);
criterion_group!(xor, xor_benchmark);
criterion_group!(geom_distr, geom_distr_benchmark);
criterion_group!(sample_exchange, sample_exchange_benchmark);

criterion_main!(sample_exchange);
