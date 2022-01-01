# Genetic Search of Diffusion Parameters

This program requires Rust's `nightly` channel! Moreover, we try to avoid heap-allocation as much as possible, which means that all parameters are hard-coded in `src/globals.rs`. Changing any one of them requires recompiling the program. As a consequence of trying to keep everything on the stack, it is necessary to increase the default stack size. Otherwise, `serde` will (probably) cause a stack overflow when trying to deserialise a previously stored state.

The easiest is to just execute the `run` script in the root directory.

## Profiling
To make profiling easier, there is a `profile` feature. When enabled, we only go through a single generation (evaluating it and performing the tournament selection) in the main thread and print the best candidate found. This makes it easier to analyse performance bottlenecks with standard tools such as flamegraph and valgrind. To build it, simply use

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --features profile
```

(the release profile already includes debug info). Afterwards, one can use e.g. cachegrind like so:

```bash
valgrind --tool=cachegrind --branch-sim=yes --cachegrind-out-file=cachegrind.out target/release/diffusion
```

Getting a flamegraph is even easier:

```bash
cargo flamegraph --bin diffusion --features profile
```

