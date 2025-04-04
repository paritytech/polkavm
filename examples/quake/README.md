# Quake for PolkaVM

This is a port of Quake which runs under PolkaVM.

You can find the source code of the guest program [here](https://github.com/koute/polkaports).

This uses the Quake shareware v1.06 PAK file.

## Running on Linux

Make sure to have SDL2 installed, and then run:

```
cargo run --release --no-default-features
```

## Running on other operating systems

It will run, but it will use an interpreter, which at this moment is *very* slow and won't run full speed.
