use criterion::{criterion_group, criterion_main, Bencher, Criterion};
use polkavm::_for_testing::PageSet;
use std::hint::black_box;
use std::time::{Duration, Instant};

const SEED: u128 = 9324658635124;
const MAX: u64 = 1048576;

fn run_insert_test<const REMOVE: bool, const CHECK: bool>(b: &mut Bencher, max_range: u64) {
    b.iter_custom(|iters| {
        let mut elapsed = Duration::default();
        for _ in 0..iters {
            let mut set = PageSet::new();
            let mut rng = oorandom::Rand64::new(SEED);
            let start = Instant::now();
            for n in 0..3000000 {
                let min = rng.rand_range(0..max_range);
                let max = rng.rand_range(min..max_range);
                let min = min as u32;
                let max = max as u32;

                if (n % 3) == 0 {
                    set.insert((min, max));
                } else if (n % 3) == 1 {
                    if REMOVE {
                        set.remove((min, max));
                    }
                } else {
                    if CHECK {
                        black_box(set.contains((min, max)));
                    }
                }
            }
            elapsed += start.elapsed();
            black_box(set);
        }

        elapsed
    });
}

fn pageset_benchmarks(c: &mut Criterion) {
    c.bench_function("insert million entries (narrow range)", |b| {
        run_insert_test::<false, false>(b, 4000)
    });
    c.bench_function("insert and remove million entries (narrow range)", |b| {
        run_insert_test::<true, false>(b, 4000)
    });
    c.bench_function("insert million entries (wide range)", |b| run_insert_test::<false, false>(b, MAX));
    c.bench_function("insert and remove million entries (wide range)", |b| {
        run_insert_test::<true, false>(b, MAX)
    });
    c.bench_function("insert, remove and check million entries (wide range)", |b| {
        run_insert_test::<true, true>(b, MAX)
    });
    c.bench_function("clear empty set", |b| {
        b.iter_custom(|iters| {
            let mut set = PageSet::new();
            let start = Instant::now();
            for _ in 0..iters {
                set.clear();
            }
            let elapsed = start.elapsed();
            black_box(set);
            elapsed
        });
    });

    c.bench_function("clear set with 50 thousand entries", |b| {
        b.iter_custom(|iters| {
            let mut elapsed = Duration::default();
            for _ in 0..iters {
                let mut set = PageSet::new();
                let mut rng = oorandom::Rand64::new(SEED);
                for _ in 0..50000 {
                    let min = rng.rand_range(0..MAX);
                    let max = rng.rand_range(min..MAX);
                    set.insert((min as u32, max as u32));
                }

                let start = Instant::now();
                set.clear();
                elapsed += start.elapsed();
                black_box(set);
            }
            elapsed
        });
    });
}

criterion_group!(benches, pageset_benchmarks);
criterion_main!(benches);
