#[polkavm_derive::polkavm_import]
extern "C" {
    // general
    #[polkavm_import(index = 0)]
    pub fn gas() -> u64;

    #[polkavm_import(index = 1)]
    pub fn lookup(s: u64, h: u64, o: u64, f: u64, l: u64) -> u64;

    #[polkavm_import(index = 2)]
    pub fn read(s: u64, ko: u64, kz: u64, o: u64, f: u64, l: u64) -> u64;

    #[polkavm_import(index = 3)]
    pub fn write(ko: u64, kz: u64, vo: u64, vz: u64) -> u64;

    #[polkavm_import(index = 4)]
    pub fn info(s: u64, o: u64) -> u64;

    // accumulate
    #[polkavm_import(index = 5)]
    pub fn bless(m: u64, a: u64, v: u64, o: u64, n: u64) -> u64;

    #[polkavm_import(index = 6)]
    pub fn assign(c: u64, o: u64) -> u64;

    #[polkavm_import(index = 7)]
    pub fn designate(o: u64) -> u64;

    #[polkavm_import(index = 8)]
    pub fn checkpoint() -> u64;

    #[polkavm_import(index = 9)]
    pub fn new(o: u64, l: u64, g: u64, m: u64) -> u64;

    #[polkavm_import(index = 10)]
    pub fn upgrade(o: u64, g: u64, m: u64) -> u64;

    #[polkavm_import(index = 11)]
    pub fn transfer(d: u64, a: u64, l: u64, o: u64) -> u64;

    #[polkavm_import(index = 12)]
    pub fn eject(d: u64, o: u64) -> u64;

    #[polkavm_import(index = 13)]
    pub fn query(o: u64, z: u64) -> u64;

    #[polkavm_import(index = 14)]
    pub fn solicit(o: u64, z: u64) -> u64;

    #[polkavm_import(index = 15)]
    pub fn forget(o: u64, z: u64) -> u64;

    #[polkavm_import(index = 16)]
    pub fn oyield(o: u64) -> u64;

    // refine
    #[polkavm_import(index = 17)]
    pub fn historical_lookup(s: u64, h: u64, o: u64, f: u64, l: u64) -> u64;

    #[polkavm_import(index = 18)]
    pub fn fetch(o: u64, f: u64, l: u64, d_type: u64, index_0: u64, index_1: u64) -> u64;

    #[polkavm_import(index = 19)]
    pub fn export(p: u64, z: u64) -> u64;

    #[polkavm_import(index = 20)]
    pub fn machine(po: u64, pz: u64, i: u64) -> u64;

    #[polkavm_import(index = 21)]
    pub fn peek(n: u64, o: u64, s: u64, z: u64) -> u64;

    #[polkavm_import(index = 22)]
    pub fn poke(n: u64, s: u64, o: u64, z: u64) -> u64;

    #[polkavm_import(index = 23)]
    pub fn zero(n: u64, p: u64, c: u64) -> u64;

    #[polkavm_import(index = 24)]
    pub fn void(n: u64, p: u64, c: u64) -> u64;

    #[polkavm_import(index = 25)]
    pub fn invoke(n: u64, o: u64) -> u64;

    #[polkavm_import(index = 26)]
    pub fn expunge(n: u64) -> u64;

    #[polkavm_import(index = 100)]
    pub fn log(level: u64, target: u64, target_len: u64, message: u64, message_len: u64) -> u64; //https://hackmd.io/@polkadot/jip1
}
