pub const GENERATION_SIZE: u32 = 512;  // Must be a multiple of 4
pub const CROSSOVER_BITS: u32 = 32;
pub const MUTATION_ODDS: u32 = 8;
pub const ELITISM: u32 = 50;

pub const N_SAMPLES: u32 = 1024;
pub const N_ROUNDS: u32 = 10;
/** Number of generations after which we exchange the samples. Must be a power of 2. */
pub const SAMPLE_LIFETIME: u32 = 8;
pub const SAMPLE_LIFETIME_MASK: u32 = SAMPLE_LIFETIME - 1;

pub const N_THREADS: u32 = 8;
pub const N_GENERATIONS: u32 = 1000;
