#![allow(clippy::exit)]
#![allow(clippy::print_stdout)]
#![allow(clippy::print_stderr)]
#![allow(clippy::use_debug)]

use clap::Parser;

mod generate;
mod test;

#[derive(Parser, Debug)]
#[clap(version)]
enum Args {
    Generate,
    Test,
}

fn main() {
    env_logger::init();

    let args = Args::parse();
    match args {
        Args::Generate => generate::main(),
        Args::Test => test::main(),
    }
}

struct Testcase {
    disassembly: String,
    json: TestcaseJson,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
struct Page {
    address: u32,
    length: u32,
    is_writable: bool,
}

#[derive(PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
struct MemoryChunk {
    address: u32,
    contents: Vec<u8>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
struct TestcaseJson {
    name: String,
    initial_regs: [u64; 13],
    initial_pc: u32,
    initial_page_map: Vec<Page>,
    initial_memory: Vec<MemoryChunk>,
    initial_gas: i64,
    program: Vec<u8>,
    expected_status: String,
    expected_regs: Vec<u64>,
    expected_pc: u32,
    expected_memory: Vec<MemoryChunk>,
    expected_gas: i64,
}
