#![forbid(non_ascii_idents)]
#![feature(stdsimd)]
#![feature(const_fn_floating_point_arithmetic)]
#[macro_use] extern crate cfg_if;
#[macro_use] extern crate static_assertions;
extern crate serde;
#[macro_use] extern crate serde_derive;

pub mod globals;
pub mod utils;
pub mod evaluation;
pub mod diffusion;
pub mod evolution;
