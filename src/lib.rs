#![feature(stdsimd)]
#![feature(const_fn_floating_point_arithmetic)]
extern crate serde;
#[macro_use] extern crate serde_derive;

pub mod globals;
pub mod utils;
pub mod diffusion;
pub mod mrxsm;
pub mod generation;
pub mod evolution;
pub mod wasserstein;
pub mod bitflips;
