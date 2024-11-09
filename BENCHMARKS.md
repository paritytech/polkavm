# Benchmark: pinky

This benchmark is a cycle-accurate NES emulator, running a real, homebrew NES game. ([source code](https://github.com/koute/polkavm/blob/9e7eba782ad094b0f465dcd375a48781ef661f55
/guest-programs/bench-pinky/src/main.rs))

## Oneshot execution (for pinky)

These benchmarks measure the end-to-end time that it takes to run the program a single time, including compilation and initialization.

| VM                                       |                 Time | vs fastest |
|------------------------------------------|----------------------|------------|
| (bare metal)                             |  33.007ms ±  0.014ms |      1.00x |
| **PolkaVM (64-bit, recompiler)**         |  55.408ms ±  0.051ms |      1.68x |
| PolkaVM (32-bit, recompiler)             |  57.264ms ±  0.025ms |      1.73x |
| PolkaVM (32-bit, recompiler, async gas)  |  63.773ms ±  0.878ms |      1.93x |
| PolkaVM (64-bit, recompiler, async gas)  |  63.897ms ±  0.099ms |      1.94x |
| PolkaVM (32-bit, recompiler, sync gas)   |  69.818ms ±  0.187ms |      2.12x |
| PolkaVM (64-bit, recompiler, sync gas)   |  70.208ms ±  0.755ms |      2.13x |
| Wasmtime (cranelift)                     | 116.002ms ±  0.329ms |      3.51x |
| Wasmtime (winch)                         | 117.121ms ±  2.528ms |      3.55x |
| Wasmtime (cranelift, epoch interruption) | 132.056ms ±  0.467ms |      4.00x |
| Wasmer (singlepass)                      | 137.832ms ±  0.287ms |      4.18x |
| Wasmtime (cranelift, fuel metering)      | 161.383ms ±  0.569ms |      4.89x |
| Solana RBPF                              | 789.195ms ±  6.903ms |     23.91x |
| Wasm3                                    | 884.142ms ±  1.092ms |     26.79x |
| Wasmi (lazy translation, checked)        |   1.021 s ±   0.001s |     30.92x |
| Wasmi (eager, checked)                   |   1.022 s ±   0.002s |     30.96x |
| Wasmi (lazy, checked)                    |   1.023 s ±   0.001s |     30.99x |
| CKB VM (ASM)                             |   1.357 s ±   0.001s |     41.11x |
| PolkaVM (64-bit, interpreter)            |   1.689 s ±   0.002s |     51.19x |
| PolkaVM (32-bit, interpreter)            |   1.877 s ±   0.002s |     56.87x |
| CKB VM (non-ASM)                         |  10.006 s ±   0.014s |    303.14x |

## Execution time (for pinky)

These benchmarks measure the execution time of the benchmark, *without* the time it takes to compile or initialize it.

| VM                                       |                 Time | vs fastest |
|------------------------------------------|----------------------|------------|
| (bare metal)                             |   3.714ms ±  0.004ms |      1.00x |
| Wasmtime (cranelift)                     |   5.776ms ±  0.006ms |      1.56x |
| **PolkaVM (64-bit, recompiler)**         |   6.422ms ±  0.156ms |      1.73x |
| PolkaVM (32-bit, recompiler)             |   6.576ms ±  0.005ms |      1.77x |
| Wasmtime (cranelift, epoch interruption) |   6.577ms ±  0.022ms |      1.77x |
| PolkaVM (32-bit, recompiler, async gas)  |   7.177ms ±  0.002ms |      1.93x |
| PolkaVM (64-bit, recompiler, async gas)  |   7.345ms ±  0.148ms |      1.98x |
| PolkaVM (64-bit, recompiler, sync gas)   |   7.957ms ±  0.083ms |      2.14x |
| PolkaVM (32-bit, recompiler, sync gas)   |   8.188ms ±  0.156ms |      2.20x |
| Wasmtime (cranelift, fuel metering)      |   8.324ms ±  0.072ms |      2.24x |
| Wasmtime (winch)                         |  10.773ms ±  0.086ms |      2.90x |
| Wasmer (singlepass)                      |  14.466ms ±  0.059ms |      3.89x |
| Solana RBPF                              |  80.809ms ±  0.539ms |     21.76x |
| Wasm3                                    |  96.046ms ±  0.159ms |     25.86x |
| Wasmi (lazy, checked)                    | 111.955ms ±  0.090ms |     30.14x |
| Wasmi (eager, checked)                   | 112.025ms ±  0.155ms |     30.16x |
| Wasmi (lazy translation, checked)        | 112.193ms ±  0.104ms |     30.21x |
| CKB VM (ASM)                             | 144.825ms ±  1.254ms |     38.99x |
| PolkaVM (64-bit, interpreter)            | 190.510ms ±  7.636ms |     51.29x |
| PolkaVM (32-bit, interpreter)            | 203.905ms ±  0.117ms |     54.90x |
| CKB VM (non-ASM)                         |   1.106 s ±   0.024s |    297.84x |

## Compilation time (for pinky)

These benchmarks measure the time it takes to compile a given program by the VM.

| VM                                       |                 Time | vs fastest |
|------------------------------------------|----------------------|------------|
| PolkaVM (32-bit, interpreter)            |   1.323µs ±  0.001µs |      1.00x |
| PolkaVM (64-bit, interpreter)            |   1.358µs ±  0.001µs |      1.03x |
| Wasmi (lazy, checked)                    |  89.733µs ±  2.832µs |     67.80x |
| **PolkaVM (64-bit, recompiler)**         | 167.553µs ±  0.072µs |    126.61x |
| PolkaVM (64-bit, recompiler, async gas)  | 177.937µs ±  0.067µs |    134.45x |
| PolkaVM (64-bit, recompiler, sync gas)   | 180.046µs ±  0.044µs |    136.05x |
| PolkaVM (32-bit, recompiler)             | 181.955µs ±  0.062µs |    137.49x |
| PolkaVM (32-bit, recompiler, async gas)  | 192.550µs ±  0.026µs |    145.50x |
| PolkaVM (32-bit, recompiler, sync gas)   | 194.093µs ±  0.094µs |    146.66x |
| Wasmi (lazy translation, checked)        | 384.444µs ±  4.764µs |    290.49x |
| Solana RBPF                              | 840.474µs ±  0.414µs |    635.08x |
| Wasmi (eager, checked)                   | 894.722µs ±  5.846µs |    676.07x |
| wazero                                   |   2.543ms ±  0.108ms |   1921.70x |
| Wasmer (singlepass)                      |   3.946ms ±  0.006ms |   2982.04x |
| Wasmtime (winch)                         |  16.569ms ±  0.087ms |  12519.78x |
| Wasmtime (cranelift)                     |  62.212ms ±  0.024ms |  47008.50x |
| Wasmtime (cranelift, epoch interruption) |  70.161ms ±  0.029ms |  53014.87x |
| Wasmtime (cranelift, fuel metering)      |  83.551ms ±  0.053ms |  63133.33x |


# Benchmark: prime-sieve

This benchmark is a prime sieve, searching for subsequent prime numbers. ([source code](https://github.com/koute/polkavm/tree/9e7eba782ad094b0f465dcd375a48781ef661f55
/guest-programs/bench-prime-sieve))

## Oneshot execution (for prime-sieve)

These benchmarks measure the end-to-end time that it takes to run the program a single time, including compilation and initialization.

| VM                                       |                 Time | vs fastest |
|------------------------------------------|----------------------|------------|
| (bare metal)                             |  10.932ms ±  0.020ms |      1.00x |
| **PolkaVM (64-bit, recompiler)**         |  14.390ms ±  0.031ms |      1.32x |
| PolkaVM (64-bit, recompiler, async gas)  |  17.682ms ±  0.030ms |      1.62x |
| PolkaVM (32-bit, recompiler)             |  19.469ms ±  0.061ms |      1.78x |
| PolkaVM (64-bit, recompiler, sync gas)   |  26.057ms ±  0.089ms |      2.38x |
| PolkaVM (32-bit, recompiler, async gas)  |  27.653ms ±  2.714ms |      2.53x |
| PolkaVM (32-bit, recompiler, sync gas)   |  27.803ms ±  0.052ms |      2.54x |
| Wasmtime (winch)                         |  54.295ms ±  0.013ms |      4.97x |
| Wasmer (singlepass)                      |  57.379ms ±  0.046ms |      5.25x |
| Wasmtime (cranelift, epoch interruption) | 108.176ms ±  0.097ms |      9.90x |
| Wasmtime (cranelift)                     | 125.107ms ±  0.361ms |     11.44x |
| Wasmtime (cranelift, fuel metering)      | 157.072ms ±  0.097ms |     14.37x |
| Wasm3                                    | 238.356ms ±  1.702ms |     21.80x |
| CKB VM (ASM)                             | 289.706ms ±  0.570ms |     26.50x |
| Wasmi (lazy, checked)                    | 398.243ms ±  1.191ms |     36.43x |
| Wasmi (lazy translation, checked)        | 399.351ms ±  0.858ms |     36.53x |
| Wasmi (eager, checked)                   | 399.988ms ±  0.821ms |     36.59x |
| Solana RBPF                              | 420.127ms ±  0.798ms |     38.43x |
| PolkaVM (64-bit, interpreter)            | 602.392ms ±  1.195ms |     55.10x |
| PolkaVM (32-bit, interpreter)            | 752.598ms ±  0.626ms |     68.84x |
| CKB VM (non-ASM)                         |   2.544 s ±   0.004s |    232.67x |

## Execution time (for prime-sieve)

These benchmarks measure the execution time of the benchmark, *without* the time it takes to compile or initialize it.

| VM                                       |                 Time | vs fastest |
|------------------------------------------|----------------------|------------|
| (bare metal)                             |   1.639ms ±  0.039ms |      1.00x |
| **PolkaVM (64-bit, recompiler)**         |   2.104ms ±  0.005ms |      1.28x |
| Wasmtime (cranelift)                     |   2.161ms ±  0.005ms |      1.32x |
| PolkaVM (64-bit, recompiler, async gas)  |   2.171ms ±  0.004ms |      1.33x |
| PolkaVM (64-bit, recompiler, sync gas)   |   2.193ms ±  0.004ms |      1.34x |
| PolkaVM (32-bit, recompiler)             |   2.403ms ±  0.005ms |      1.47x |
| Wasmtime (cranelift, epoch interruption) |   2.435ms ±  0.006ms |      1.49x |
| Wasmtime (cranelift, fuel metering)      |   2.465ms ±  0.010ms |      1.50x |
| PolkaVM (32-bit, recompiler, async gas)  |   2.467ms ±  0.006ms |      1.51x |
| PolkaVM (32-bit, recompiler, sync gas)   |   2.482ms ±  0.004ms |      1.51x |
| Wasmtime (winch)                         |   5.081ms ±  0.004ms |      3.10x |
| Wasmer (singlepass)                      |   6.120ms ±  0.025ms |      3.73x |
| Wasm3                                    |  22.545ms ±  0.046ms |     13.76x |
| CKB VM (ASM)                             |  27.559ms ±  0.006ms |     16.82x |
| Wasmi (lazy, checked)                    |  46.514ms ±  0.024ms |     28.38x |
| Wasmi (lazy translation, checked)        |  46.521ms ±  0.033ms |     28.39x |
| Wasmi (eager, checked)                   |  47.138ms ±  1.106ms |     28.76x |
| PolkaVM (64-bit, interpreter)            |  75.746ms ±  0.033ms |     46.22x |
| PolkaVM (32-bit, interpreter)            |  91.441ms ±  0.043ms |     55.80x |
| Solana RBPF                              | 166.772ms ±  0.511ms |    101.77x |
| CKB VM (non-ASM)                         | 374.649ms ±  0.691ms |    228.62x |

## Compilation time (for prime-sieve)

These benchmarks measure the time it takes to compile a given program by the VM.

| VM                                       |                 Time | vs fastest |
|------------------------------------------|----------------------|------------|
| PolkaVM (64-bit, interpreter)            |   0.581µs ±  0.000µs |      1.00x |
| PolkaVM (32-bit, interpreter)            |   0.582µs ±  0.000µs |      1.00x |
| Wasmi (lazy, checked)                    |  77.007µs ±  2.521µs |    132.53x |
| PolkaVM (32-bit, recompiler)             | 434.055µs ±  0.191µs |    747.03x |
| **PolkaVM (64-bit, recompiler)**         | 442.354µs ±  0.369µs |    761.31x |
| PolkaVM (32-bit, recompiler, async gas)  | 456.789µs ±  0.204µs |    786.16x |
| PolkaVM (32-bit, recompiler, sync gas)   | 464.064µs ±  0.187µs |    798.68x |
| PolkaVM (64-bit, recompiler, async gas)  | 465.916µs ±  0.201µs |    801.86x |
| PolkaVM (64-bit, recompiler, sync gas)   | 473.516µs ±  0.243µs |    814.94x |
| Wasmi (lazy translation, checked)        | 598.222µs ±  2.895µs |   1029.57x |
| Wasmi (eager, checked)                   |   1.668ms ±  0.004ms |   2870.28x |
| Solana RBPF                              |   2.252ms ±  0.001ms |   3876.33x |
| wazero                                   |   4.049ms ±  0.423ms |   6969.12x |
| Wasmtime (winch)                         |   8.510ms ±  0.004ms |  14645.59x |
| Wasmer (singlepass)                      |   9.421ms ±  0.016ms |  16213.84x |
| Wasmtime (cranelift)                     |  76.662ms ±  0.101ms | 131939.72x |
| Wasmtime (cranelift, epoch interruption) |  84.835ms ±  0.047ms | 146004.53x |
| Wasmtime (cranelift, fuel metering)      | 130.264ms ±  0.052ms | 224191.07x |


# Benchmark: minimal

This benchmark is a tiny, minimal program which doesn't do much work; it just increments a global variable and returns immediately. It is a good test case for measuring constant-time overhead. ([source code](https://github.com/koute/polkavm/blob/9e7eba782ad094b0f465dcd375a48781ef661f55
/guest-programs/bench-minimal/src/main.rs))

## Oneshot execution (for minimal)

These benchmarks measure the end-to-end time that it takes to run the program a single time, including compilation and initialization.

| VM                                       |                 Time | vs fastest |
|------------------------------------------|----------------------|------------|
| PolkaVM (64-bit, interpreter)            |   2.533µs ±  0.002µs |      1.00x |
| PolkaVM (32-bit, interpreter)            |   2.563µs ±  0.006µs |      1.01x |
| CKB VM (non-ASM)                         |   7.198µs ±  0.006µs |      2.84x |
| Wasm3                                    |  23.486µs ±  0.034µs |      9.27x |
| Wasmi (lazy, checked)                    |  25.833µs ±  0.226µs |     10.20x |
| Wasmi (lazy translation, checked)        |  26.224µs ±  0.321µs |     10.35x |
| Wasmi (eager, checked)                   |  27.169µs ±  0.442µs |     10.73x |
| Solana RBPF                              |  30.957µs ±  0.035µs |     12.22x |
| (bare metal)                             |  42.610µs ±  0.127µs |     16.82x |
| CKB VM (ASM)                             |  65.851µs ±  0.128µs |     26.00x |
| PolkaVM (32-bit, recompiler)             | 113.521µs ±  0.339µs |     44.82x |
| PolkaVM (64-bit, recompiler, sync gas)   | 113.633µs ±  0.367µs |     44.87x |
| PolkaVM (32-bit, recompiler, async gas)  | 113.685µs ±  0.302µs |     44.89x |
| **PolkaVM (64-bit, recompiler)**         | 113.743µs ±  0.298µs |     44.91x |
| PolkaVM (64-bit, recompiler, async gas)  | 113.778µs ±  0.340µs |     44.92x |
| PolkaVM (32-bit, recompiler, sync gas)   | 113.916µs ±  0.280µs |     44.98x |
| Wasmer (singlepass)                      | 128.467µs ±  1.983µs |     50.72x |
| Wasmtime (winch)                         | 644.160µs ±  1.746µs |    254.34x |
| Wasmtime (cranelift)                     | 935.241µs ±  2.305µs |    369.27x |
| Wasmtime (cranelift, epoch interruption) |   1.089ms ±  0.001ms |    430.12x |
| Wasmtime (cranelift, fuel metering)      |   1.144ms ±  0.001ms |    451.51x |

## Execution time (for minimal)

These benchmarks measure the execution time of the benchmark, *without* the time it takes to compile or initialize it.

| VM                                       |                 Time | vs fastest |
|------------------------------------------|----------------------|------------|
| (bare metal)                             |   0.050µs ±  0.003µs |      1.00x |
| Wasmer (singlepass)                      |   0.139µs ±  0.002µs |      2.79x |
| Wasm3                                    |   0.144µs ±  0.001µs |      2.90x |
| Wasmi (lazy translation, checked)        |   0.151µs ±  0.000µs |      3.03x |
| Wasmi (lazy, checked)                    |   0.152µs ±  0.000µs |      3.05x |
| Wasmi (eager, checked)                   |   0.154µs ±  0.000µs |      3.09x |
| Solana RBPF                              |   0.167µs ±  0.000µs |      3.35x |
| PolkaVM (32-bit, interpreter)            |   0.203µs ±  0.000µs |      4.07x |
| PolkaVM (64-bit, interpreter)            |   0.203µs ±  0.000µs |      4.07x |
| Wasmtime (cranelift, epoch interruption) |   0.247µs ±  0.004µs |      4.96x |
| Wasmtime (winch)                         |   0.249µs ±  0.003µs |      4.99x |
| Wasmtime (cranelift, fuel metering)      |   0.250µs ±  0.003µs |      5.02x |
| Wasmtime (cranelift)                     |   0.252µs ±  0.006µs |      5.06x |
| CKB VM (ASM)                             |   2.708µs ±  0.002µs |     54.35x |
| CKB VM (non-ASM)                         |   3.501µs ±  0.001µs |     70.28x |
| **PolkaVM (64-bit, recompiler)**         |   4.852µs ±  0.040µs |     97.39x |
| PolkaVM (32-bit, recompiler, async gas)  |   4.893µs ±  0.049µs |     98.21x |
| PolkaVM (32-bit, recompiler)             |   4.918µs ±  0.034µs |     98.72x |
| PolkaVM (64-bit, recompiler, sync gas)   |   4.929µs ±  0.024µs |     98.94x |
| PolkaVM (64-bit, recompiler, async gas)  |   4.937µs ±  0.031µs |     99.09x |
| PolkaVM (32-bit, recompiler, sync gas)   |   4.943µs ±  0.046µs |     99.22x |

## Compilation time (for minimal)

These benchmarks measure the time it takes to compile a given program by the VM.

| VM                                       |                 Time | vs fastest |
|------------------------------------------|----------------------|------------|
| PolkaVM (64-bit, interpreter)            |   0.589µs ±  0.001µs |      1.00x |
| PolkaVM (32-bit, interpreter)            |   0.599µs ±  0.001µs |      1.02x |
| PolkaVM (32-bit, recompiler)             |   1.810µs ±  0.001µs |      3.07x |
| **PolkaVM (64-bit, recompiler)**         |   1.823µs ±  0.002µs |      3.09x |
| PolkaVM (32-bit, recompiler, async gas)  |   1.830µs ±  0.001µs |      3.11x |
| PolkaVM (64-bit, recompiler, async gas)  |   1.832µs ±  0.001µs |      3.11x |
| PolkaVM (64-bit, recompiler, sync gas)   |   1.852µs ±  0.002µs |      3.14x |
| PolkaVM (32-bit, recompiler, sync gas)   |   1.854µs ±  0.002µs |      3.15x |
| Wasmi (lazy, checked)                    |   5.734µs ±  0.074µs |      9.73x |
| Wasmi (lazy translation, checked)        |   6.572µs ±  0.112µs |     11.15x |
| Wasmi (eager, checked)                   |   8.331µs ±  0.020µs |     14.14x |
| Solana RBPF                              |  25.867µs ±  0.034µs |     43.90x |
| Wasmer (singlepass)                      |  85.955µs ±  5.566µs |    145.87x |
| wazero                                   | 197.802µs ±  0.387µs |    335.69x |
| Wasmtime (winch)                         | 575.089µs ±  2.634µs |    975.98x |
| Wasmtime (cranelift)                     | 858.096µs ±  9.056µs |   1456.27x |
| Wasmtime (cranelift, epoch interruption) |   1.021ms ±  0.004ms |   1732.84x |
| Wasmtime (cranelift, fuel metering)      |   1.068ms ±  0.010ms |   1812.18x |


---------------------------------------------------------------------------

# Supplemental information

CPU: AMD Ryzen Threadripper 3970X 32-Core Processor

Platform: x86_64-linux

Commit: [9e7eba782ad094b0f465dcd375a48781ef661f55
](https://github.com/koute/polkavm/tree/9e7eba782ad094b0f465dcd375a48781ef661f55
)

Timestamp: 2024-11-24 03:45:40 UTC

---------------------------------------------------------------------------

# Replication

You can replicate these benchmarks as follows:

```
$ git clone https://github.com/koute/polkavm.git
$ cd polkavm
$ git checkout 9e7eba782ad094b0f465dcd375a48781ef661f55

$ cd tools/benchtool
$ ./01-build-benchmarks.sh
$ ./02-run-benchmarks.rb
$ ./03-analyze-benchmarks.rb
```

Only running the benchmarks on Linux is officially supported.

WARNING: The `02-run-benchmarks.rb` script uses a couple of system-level tricks to make benchmarking more consistent and requires 'sudo' and 'schedtool' to be installed. If you're uncomfortable with that or if you're running a non-Linux OS you can also run the benchmarks with `cargo run --release` instead.

