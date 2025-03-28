#![no_std]
#![no_main]
#![feature(asm_const)]

extern crate alloc;
use alloc::string::String;
use alloc::format;
use alloc::vec;

use polkavm_derive::min_stack_size;
min_stack_size!(409600);

use utils::constants::{FIRST_READABLE_ADDRESS, FIRST_READABLE_PAGE, NONE, SEGMENT_SIZE};
use utils::functions::{parse_accumulate_args, parse_refine_args, call_log};
use utils::host_functions::{export, fetch};
use utils::hash_functions::blake2b_hash;

const PAGE_SIZE: usize = 4096;
const PAGE_DIM: usize = 64;
const NUM_PAGES: usize = 9;
const PAGES_PER_ROW: usize = 3;
const TOTAL_ROWS: usize = PAGE_DIM * 3;
const TOTAL_COLS: usize = PAGE_DIM * 3;

const TOTAL_BYTES: usize = PAGE_SIZE * NUM_PAGES;

const ROWS_WITH_GHOST: usize = TOTAL_ROWS + 2;
const COLS_WITH_GHOST: usize = TOTAL_COLS + 2;

/* glider pattern (3×3):
   row=0: [0,1,0]
   row=1: [0,0,1]
   row=2: [1,1,1]
*/
const GLIDER_PATTERN: [(usize, usize); 5] = [
    (0, 1),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
];

const H_SPACING: usize = 5;
const V_SPACING: usize = 5;
const MARGIN: usize = 2;
const GLIDERS_PER_ROW: usize = (PAGE_DIM - MARGIN) / H_SPACING;

fn bytes_to_hex(data: &[u8]) -> String {
    data.iter().map(|b| format!("{:02x}", b)).collect()
}

struct Grid {
    pages: [[u8; PAGE_SIZE]; NUM_PAGES],
}

impl Grid {
    fn new() -> Self {
        Self {
            pages: [[0; PAGE_SIZE]; NUM_PAGES],
        }
    }

    fn init_hash_chain(&mut self) {
        // let num_hashes = TOTAL_BYTES / 32;
        let num_hashes = 20;
        let hash_data_size = 32 * num_hashes;
        let mut hash_data = vec![0u8; hash_data_size];
        let mut current = String::new();
        let mut first_hash = [0u8; 32];

        for i in 0..num_hashes {
            let hash_bytes = blake2b_hash(current.as_bytes());
            if i == 0 {
                first_hash.copy_from_slice(&hash_bytes);
            }
            hash_data[i * 32..(i + 1) * 32].copy_from_slice(&hash_bytes);
            current = bytes_to_hex(&hash_bytes);
        }
    
        let mut hash_chain = [0u8; TOTAL_BYTES];
        for (i, byte) in hash_chain.iter_mut().enumerate() {
            *byte = hash_data[i % hash_data_size];
            *byte >>= 7;
        }

        call_log(
            2,
            None,
            &format!(
                "Finish {:?} hashes and shift, first hash: {:?}",
                num_hashes,
                bytes_to_hex(&first_hash)
            ),
        );

        for i in 0..NUM_PAGES {
            let start = i * PAGE_SIZE;
            self.pages[i].copy_from_slice(&hash_chain[start..start + PAGE_SIZE]);
        }

        call_log(2, None, &format!("Finish hash chain copy"));
    }

    fn init_gliders(&mut self, num_gliders: usize) {
        self.pages = [[0; PAGE_SIZE]; NUM_PAGES];

        let mut placed_count = 0;
        for i in 0..num_gliders {
            let row_index = i / GLIDERS_PER_ROW;
            let col_index = i % GLIDERS_PER_ROW;

            let row_offset = row_index * V_SPACING + MARGIN;
            let col_offset = col_index * H_SPACING + MARGIN;

            if row_offset + 2 >= PAGE_DIM || col_offset + 2 >= PAGE_DIM {
                break;
            }

            for &(r, c) in &GLIDER_PATTERN {
                let row = row_offset + r;
                let col = col_offset + c;
                self.pages[0][row * PAGE_DIM + col] = 1;
            }
            placed_count += 1;
        }
        call_log(2, None, &format!("Finish glider init, placed {} gliders", placed_count));
    }

    fn step(&self) -> Grid {
        let mut in_buf: [[u8; COLS_WITH_GHOST]; ROWS_WITH_GHOST] =
            [[0; COLS_WITH_GHOST]; ROWS_WITH_GHOST];

        for page in 0..NUM_PAGES {
            let page_row = page / PAGES_PER_ROW;
            let page_col = page % PAGES_PER_ROW;
            for i in 0..PAGE_DIM {
                for j in 0..PAGE_DIM {
                    let global_row = page_row * PAGE_DIM + i;
                    let global_col = page_col * PAGE_DIM + j;
                    in_buf[global_row + 1][global_col + 1] =
                        self.pages[page][i * PAGE_DIM + j];
                }
            }
        }

        // ghost border wrap-around
        for j in 1..=TOTAL_COLS {
            in_buf[0][j] = in_buf[TOTAL_ROWS][j];
            in_buf[TOTAL_ROWS + 1][j] = in_buf[1][j];
        }
        for i in 1..=TOTAL_ROWS {
            in_buf[i][0] = in_buf[i][TOTAL_COLS];
            in_buf[i][TOTAL_COLS + 1] = in_buf[i][1];
        }
        in_buf[0][0] = in_buf[TOTAL_ROWS][TOTAL_COLS];
        in_buf[0][TOTAL_COLS + 1] = in_buf[TOTAL_ROWS][1];
        in_buf[TOTAL_ROWS + 1][0] = in_buf[1][TOTAL_COLS];
        in_buf[TOTAL_ROWS + 1][TOTAL_COLS + 1] = in_buf[1][1];

        let mut out_buf: [[u8; TOTAL_COLS]; TOTAL_ROWS] = [[0; TOTAL_COLS]; TOTAL_ROWS];
        for i in 1..=TOTAL_ROWS {
            for j in 1..=TOTAL_COLS {
                let live_neighbors =
                    in_buf[i - 1][j - 1] as u32 +
                    in_buf[i - 1][j] as u32 +
                    in_buf[i - 1][j + 1] as u32 +
                    in_buf[i][j - 1] as u32 +
                    in_buf[i][j + 1] as u32 +
                    in_buf[i + 1][j - 1] as u32 +
                    in_buf[i + 1][j] as u32 +
                    in_buf[i + 1][j + 1] as u32;
                let current = in_buf[i][j];
                out_buf[i - 1][j - 1] = if (current == 1 && (live_neighbors == 2 || live_neighbors == 3))
                    || (current == 0 && live_neighbors == 3)
                {
                    1
                } else {
                    0
                };
            }
        }

        let mut new_pages = [[0u8; PAGE_SIZE]; NUM_PAGES];
        for page in 0..NUM_PAGES {
            let page_row = page / PAGES_PER_ROW;
            let page_col = page % PAGES_PER_ROW;
            for i in 0..PAGE_DIM {
                for j in 0..PAGE_DIM {
                    let global_row = page_row * PAGE_DIM + i;
                    let global_col = page_col * PAGE_DIM + j;
                    new_pages[page][i * PAGE_DIM + j] = out_buf[global_row][global_col];
                }
            }
        }

        Grid { pages: new_pages }
    }
}

#[polkavm_derive::polkavm_export]
extern "C" fn refine(start_address: u64, length: u64) -> (u64, u64) {
    let (_wi_service_index, wi_payload_start_address, _wi_payload_length, _wphash) =
        if let Some(args) = parse_refine_args(start_address, length) {
            (
                args.wi_service_index,
                args.wi_payload_start_address,
                args.wi_payload_length,
                args.wphash,
            )
        } else {
            return (FIRST_READABLE_ADDRESS as u64, 0);
        };

    let mut grid = Grid::new();

    let step_n: u32 = unsafe { (*(wi_payload_start_address as *const u32)).into() };

    let num_of_gloders_address: u64 = wi_payload_start_address + 4;
    let num_of_gloders: u32 = unsafe { (*(num_of_gloders_address as *const u32)).into() };

    let total_execution_steps_address: u64 = wi_payload_start_address + 8;
    let total_execution_steps: u32 = unsafe { (*(total_execution_steps_address as *const u32)).into() };

    call_log(2, None, &format!("Step_n: {:?}, gliders: {:?}, total_steps: {:?}", step_n, num_of_gloders, total_execution_steps));

    let mut seg_cnt = 0;
    for i in 0..NUM_PAGES {
        let page_address = grid.pages[i].as_ptr() as u64;
        let page_length = PAGE_SIZE as u64;
        let fetch_result = unsafe { fetch(page_address, 0, page_length, 6, i as u64, 0) };
        if fetch_result == NONE {
            break;
        }
        seg_cnt += 1;
    }

    if seg_cnt > 0 {
        call_log(2, None, &format!("Step_n: {:?}, Fetched {:?} segments", step_n, seg_cnt));
    } else {
        grid.init_gliders(num_of_gloders as usize);
    }

    for _ in 0..total_execution_steps {
        grid = grid.step();
    }
    
    /*
    let total_cells = TOTAL_ROWS * TOTAL_COLS;
    let poke_index = (steps as usize) % total_cells;
    let poke_row = poke_index / TOTAL_COLS;
    let poke_col = poke_index % TOTAL_COLS;
    let page_row = poke_row / PAGE_DIM;
    let page_col = poke_col / PAGE_DIM;
    let page_index = page_row * PAGES_PER_ROW + page_col;
    let local_row = poke_row % PAGE_DIM;
    let local_col = poke_col % PAGE_DIM;
    grid.pages[page_index][local_row * PAGE_DIM + local_col] = 1;
    */

    for i in 0..NUM_PAGES {
        let page_address = grid.pages[i].as_ptr() as u64;
        let page_length = PAGE_SIZE as u64;
        unsafe { export(page_address, page_length); }
    }

    call_log(2, None, &format!("Step_n: {:?}, exported {:?} segments, done", step_n, NUM_PAGES));

    return (FIRST_READABLE_ADDRESS as u64, 0);
}

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate(start_address: u64, length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0);
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer(_start_address: u64, _length: u64) -> (u64, u64) {
    return (FIRST_READABLE_ADDRESS as u64, 0);
}
