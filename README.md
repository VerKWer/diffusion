This program requires Rust's `nightly` channel! Moreover, we try to avoid heap-allocation as much as possible, which means that all parameters are hard-coded in `src/globals.rs`. Changing any one of them requires recompiling the program. As a consequence of trying to keep everything on the stack, it is necessary to increase the default stack size. Otherwise, `serde` will (probably) cause a stack overflow when trying to deserialise a previously stored state.

The easiest is to just execute the `run` script in this directory.