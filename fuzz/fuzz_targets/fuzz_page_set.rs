#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use polkavm::_for_testing::PageSet;

#[derive(Arbitrary, Debug)]
enum Action {
    Insert { min: u8, max: u8 },
    Remove { min: u8, max: u8 },
    Contains { min: u8, max: u8 },
}

#[derive(Arbitrary, Debug)]
struct Input {
    actions: Vec<Action>,
}

#[derive(Default)]
struct PageSetNaive {
    bitmask: [u64; 4],
}

impl PageSetNaive {
    fn insert_one(&mut self, n: u8) {
        self.bitmask[usize::from(n >> 6)] |= 1 << (n & 0b00111111);
    }

    fn remove_one(&mut self, n: u8) {
        self.bitmask[usize::from(n >> 6)] &= !(1 << (n & 0b00111111));
    }

    fn contains_one(&self, n: u8) -> bool {
        (self.bitmask[usize::from(n >> 6)] & (1 << (n & 0b00111111))) != 0
    }

    fn insert(&mut self, min: u8, max: u8) {
        for n in min..=max {
            self.insert_one(n);
        }
    }

    fn remove(&mut self, min: u8, max: u8) {
        for n in min..=max {
            self.remove_one(n);
        }
    }

    fn contains(&self, min: u8, max: u8) -> bool {
        for n in min..=max {
            if !self.contains_one(n) {
                return false;
            }
        }
        true
    }
}

fn swap(min: &mut u8, max: &mut u8) {
    if *min > *max {
        let max_value = *max;
        *max = *min;
        *min = max_value;
    }
}

fuzz_target!(|input: Input| {
    let mut page_set = PageSet::new();
    let mut naive = PageSetNaive::default();
    for action in input.actions {
        match action {
            Action::Insert { mut min, mut max } => {
                swap(&mut min, &mut max);
                page_set.insert((u32::from(min), u32::from(max)));
                naive.insert(min, max);
            }
            Action::Remove { mut min, mut max } => {
                swap(&mut min, &mut max);
                page_set.remove((u32::from(min), u32::from(max)));
                naive.remove(min, max);
            }
            Action::Contains { mut min, mut max } => {
                swap(&mut min, &mut max);
                assert_eq!(
                    page_set.contains((u32::from(min), u32::from(max))),
                    naive.contains(min, max),
                    "contains failed for: ({min}, {max})"
                );
            }
        }
    }
});
