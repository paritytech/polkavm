#![no_std]
#![no_main]

extern crate alloc;
use simplealloc::SimpleAlloc;

#[global_allocator]
static ALLOCATOR: SimpleAlloc<4096> = SimpleAlloc::new();

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe {
        core::arch::asm!("unimp", options(noreturn));
    }
}
#[polkavm_derive::polkavm_import]
extern "C" {
    #[polkavm_import(index = 2)]
    pub fn read(service: u64, key_ptr: u64, key_len: u64, out: u64, out_len: u64) -> u64;
    #[polkavm_import(index = 3)]
    pub fn write(ko: u64, kz: u64, bo: u64, bz: u64) -> u64;
    #[polkavm_import(index = 30)]
    pub fn fetch(o: u64, l_off: u64, dataid: u64, work_item_index: u64, extrinsic_index: u64) -> u64;
    #[polkavm_import(index = 65)]
    pub fn ed25519verify(pubkey_addr: u64, data_addr: u64, data_len: u64, signature_addr: u64) -> u64;
}

pub const NONE: u64 = u64::MAX;
pub const OOB: u64 = u64::MAX - 2;

pub const SIGNATURE_LEN: usize = 64;
pub const PUBLIC_KEY_LEN: usize = 32;

pub const METHOD_ID_LEN: usize = 4;
pub const BUFFER_SIZE: usize = 512;
pub const FIRST_WRITEABLE_ADDRESS: u64 = 0xFEFDE000;

pub const ASSET_SIZE: usize = 8 + 32 + 8 + 32 + 8 + 1;
pub const ASSET_ID_SIZE: usize = 8;
pub const ACCOUNT_SIZE: usize = 8 * 3;
pub const ACCOUNT_STORAGE_KEY_SIZE: usize = 8 + 32;
#[repr(C)]
#[derive(Default, Debug, Clone)]
pub struct Asset {
    pub asset_id: u64,
    pub issuer: [u8; 32],
    pub min_balance: u64,
    pub symbol: [u8; 32],
    pub total_supply: u64,
    pub decimals: u8,
}

impl Asset {
    pub fn bytes(&self) -> [u8; ASSET_SIZE] {
        let mut result = [0u8; ASSET_SIZE];
        let mut offset = 0;

        result[offset..offset + 8].copy_from_slice(&self.asset_id.to_le_bytes());
        offset += 8;
        result[offset..offset + 32].copy_from_slice(&self.issuer);
        offset += 32;
        result[offset..offset + 8].copy_from_slice(&self.min_balance.to_le_bytes());
        offset += 8;
        result[offset..offset + 32].copy_from_slice(&self.symbol);
        offset += 32;
        result[offset..offset + 8].copy_from_slice(&self.total_supply.to_le_bytes());
        offset += 8;
        result[offset] = self.decimals;
        result
    }

    pub fn from_bytes(bytes: [u8; ASSET_SIZE]) -> Self {
        let mut offset = 0;

        let asset_id = {
            let mut temp = [0u8; 8];
            temp.copy_from_slice(&bytes[offset..offset + 8]);
            offset += 8;
            u64::from_le_bytes(temp)
        };

        let issuer = {
            let mut temp = [0u8; 32];
            temp.copy_from_slice(&bytes[offset..offset + 32]);
            offset += 32;
            temp
        };

        let min_balance = {
            let mut temp = [0u8; 8];
            temp.copy_from_slice(&bytes[offset..offset + 8]);
            offset += 8;
            u64::from_le_bytes(temp)
        };

        let symbol = {
            let mut temp = [0u8; 32];
            temp.copy_from_slice(&bytes[offset..offset + 32]);
            offset += 32;
            temp
        };

        let total_supply = {
            let mut temp = [0u8; 8];
            temp.copy_from_slice(&bytes[offset..offset + 8]);
            offset += 8;
            u64::from_le_bytes(temp)
        };

        let decimals = bytes[offset];

        Self {
            asset_id,
            issuer,
            min_balance,
            symbol,
            total_supply,
            decimals,
        }
    }

    pub fn read_from_storage(asset_id_bytes: &[u8]) -> Option<Self> {
        let mut asset_buf = [0u8; ASSET_SIZE];
        let ret = unsafe {
            read(
                NONE,
                asset_id_bytes.as_ptr() as u64,
                ASSET_ID_SIZE as u64,
                asset_buf.as_mut_ptr() as u64,
                asset_buf.len() as u64,
            )
        };
        if ret == NONE || ret == OOB {
            None
        } else {
            Some(Self::from_bytes(asset_buf))
        }
    }

    pub fn write_to_storage(&self, asset_id_bytes: &[u8]) {
        let new_asset_bytes = self.bytes();
        unsafe {
            write(
                asset_id_bytes.as_ptr() as u64,
                ASSET_ID_SIZE as u64,
                new_asset_bytes.as_ptr() as u64,
                new_asset_bytes.len() as u64,
            );
        }
    }
}

#[repr(C)]
#[derive(Default, Debug, Clone)]
pub struct Account {
    pub nonce: u64,
    pub free: u64,
    pub reserved: u64,
}

impl Account {
    pub fn bytes(&self) -> [u8; ACCOUNT_SIZE] {
        let mut result = [0u8; ACCOUNT_SIZE];
        let mut offset = 0;

        result[offset..offset + 8].copy_from_slice(&self.nonce.to_le_bytes());
        offset += 8;
        result[offset..offset + 8].copy_from_slice(&self.free.to_le_bytes());
        offset += 8;
        result[offset..offset + 8].copy_from_slice(&self.reserved.to_le_bytes());
        result
    }

    pub fn from_bytes(bytes: [u8; ACCOUNT_SIZE]) -> Self {
        let mut offset = 0;

        let nonce = {
            let mut temp = [0u8; 8];
            temp.copy_from_slice(&bytes[offset..offset + 8]);
            offset += 8;
            u64::from_le_bytes(temp)
        };
        let free = {
            let mut temp = [0u8; 8];
            temp.copy_from_slice(&bytes[offset..offset + 8]);
            offset += 8;
            u64::from_le_bytes(temp)
        };
        let reserved = {
            let mut temp = [0u8; 8];
            temp.copy_from_slice(&bytes[offset..offset + 8]);
            u64::from_le_bytes(temp)
        };
        Self {
            nonce,
            free,
            reserved,
        }
    }

    pub fn read_from_storage(asset_id_bytes: &[u8], account_id_bytes: &[u8]) -> Option<Self> {
        let mut account_key = [0u8; ACCOUNT_STORAGE_KEY_SIZE];
        account_key[0..8].copy_from_slice(asset_id_bytes);
        account_key[8..40].copy_from_slice(account_id_bytes);

        let mut account_buf = [0u8; ACCOUNT_SIZE];
        let ret = unsafe {
            read(
                NONE,
                account_key.as_ptr() as u64,
                account_key.len() as u64,
                account_buf.as_mut_ptr() as u64,
                account_buf.len() as u64,
            )
        };
        if ret == NONE || ret == OOB {
            None
        } else {
            Some(Self::from_bytes(account_buf))
        }
    }

    pub fn write_to_storage(&self, asset_id_bytes: &[u8], account_id_bytes: &[u8]) {
        let mut account_key = [0u8; ACCOUNT_STORAGE_KEY_SIZE];
        account_key[0..8].copy_from_slice(asset_id_bytes);
        account_key[8..40].copy_from_slice(account_id_bytes);

        let new_account_bytes = self.bytes();
        unsafe {
            write(
                account_key.as_ptr() as u64,
                account_key.len() as u64,
                new_account_bytes.as_ptr() as u64,
                new_account_bytes.len() as u64,
            );
        }
    }
}

#[polkavm_derive::polkavm_export]
extern "C" fn is_authorized() -> u32 {
    0
}

#[polkavm_derive::polkavm_export]
extern "C" fn refine() -> u32 {
    let n: u64 = unsafe { ( *(0xFEFF0004 as *const u64)).into() }; // get extrinsic index from payload

    let extrinsic_len = unsafe { fetch(FIRST_WRITEABLE_ADDRESS, BUFFER_SIZE.try_into().unwrap(), 17, n as u64, 0) } as usize;
    let pubkey_ptr = FIRST_WRITEABLE_ADDRESS as *const u8;

    let data_ptr = unsafe { pubkey_ptr.add(PUBLIC_KEY_LEN) };
    let data_len = extrinsic_len - PUBLIC_KEY_LEN - SIGNATURE_LEN;

    let sig_ptr = unsafe { pubkey_ptr.add(extrinsic_len - SIGNATURE_LEN) };

    let is_valid = unsafe {
        ed25519verify(
            pubkey_ptr as u64,
            data_ptr as u64,
            data_len as u64,
            sig_ptr as u64,
        )
    };

    if is_valid == 0 {
        unsafe {
            core::arch::asm!(
                "mv a3, {0}",
                "mv a4, {1}",
                in(reg) data_ptr,
                in(reg) data_len,
            );
        }
    }
    0
}

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate() -> u32 {
    let extrinsic_address: u64;
    let extrinsic_len: u64;

    unsafe {
        core::arch::asm!(
            "mv {0}, a0",
            "mv {1}, a1",
            out(reg) extrinsic_address,
            out(reg) extrinsic_len,
        );
    }

    let extrinsic: &[u8] = unsafe { core::slice::from_raw_parts(extrinsic_address as *const u8, extrinsic_len as usize) };

    let method_id = &extrinsic[..METHOD_ID_LEN];
    let payload = &extrinsic[METHOD_ID_LEN..];

    match extract_u32(method_id) {
        0 => create_asset(payload),
        1 => mint(payload),
        2 => burn(payload),
        3 => bond(payload),
        4 => unbond(payload),
        5 => transfer(payload),
        _ => {}
    }
    0
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer() -> u32 {
    0
}

pub fn create_asset(payload: &[u8]) {
    let asset_id = extract_u64(&payload[0..8]);
    unsafe {
        write(
            &asset_id as *const u64 as u64,
            ASSET_ID_SIZE as u64,
            payload.as_ptr() as u64,
            payload.len() as u64,
        )
    };
}

pub fn mint(payload: &[u8]) {
    let asset_id_bytes = &payload[0..8];
    let account_id_bytes = &payload[8..40];
    let amount = extract_u64(&payload[40..48]);

    let Some(mut asset) = Asset::read_from_storage(asset_id_bytes) else { return };
    asset.total_supply = asset.total_supply.saturating_add(amount);
    asset.write_to_storage(asset_id_bytes);

    let mut account = match Account::read_from_storage(asset_id_bytes, account_id_bytes) {
        Some(a) => a,
        None => Account::default(),
    };
    account.nonce += 1;
    account.free = account.free.saturating_add(amount);
    account.write_to_storage(asset_id_bytes, account_id_bytes);
}

pub fn burn(payload: &[u8]) {
    let asset_id_bytes = &payload[0..8];
    let account_id_bytes = &payload[8..40];
    let amount = extract_u64(&payload[40..48]);

    let Some(mut asset) = Asset::read_from_storage(asset_id_bytes) else { return };
    if asset.total_supply < amount {
        return;
    }
    asset.total_supply -= amount;
    asset.write_to_storage(asset_id_bytes);

    let Some(mut account) = Account::read_from_storage(asset_id_bytes, account_id_bytes) else { return };
    if account.free < amount {
        return;
    }

    account.nonce += 1;
    account.free -= amount;
    account.write_to_storage(asset_id_bytes, account_id_bytes);
}

pub fn bond(payload: &[u8]) {
    let asset_id_bytes = &payload[0..8];
    let account_id_bytes = &payload[8..40];
    let amount = extract_u64(&payload[40..48]);

    let Some(mut account) = Account::read_from_storage(asset_id_bytes, account_id_bytes) else { return };
    if account.free < amount {
        return;
    }
    account.nonce += 1;
    account.free -= amount;
    account.reserved += amount;
    account.write_to_storage(asset_id_bytes, account_id_bytes);
}

pub fn unbond(payload: &[u8]) {
    let asset_id_bytes = &payload[0..8];
    let account_id_bytes = &payload[8..40];
    let amount = extract_u64(&payload[40..48]);

    let Some(mut account) = Account::read_from_storage(asset_id_bytes, account_id_bytes) else { return };
    if account.reserved < amount {
        return;
    }

    account.nonce += 1;
    account.free += amount;
    account.reserved -= amount;
    account.write_to_storage(asset_id_bytes, account_id_bytes);
}

pub fn transfer(payload: &[u8]) {
    let asset_id_bytes = &payload[0..8];
    let sender_account_bytes = &payload[8..40];
    let receiver_account_bytes = &payload[40..72];
    let amount = extract_u64(&payload[72..80]);

    let Some(mut sender_account) = Account::read_from_storage(asset_id_bytes, sender_account_bytes) else { return };
    if sender_account.free < amount {
        return;
    }

    sender_account.nonce += 1;
    sender_account.free -= amount;
    sender_account.write_to_storage(asset_id_bytes, sender_account_bytes);

    let mut receiver_account = match Account::read_from_storage(asset_id_bytes, receiver_account_bytes) {
        Some(a) => a,
        None => Account::default(),
    };
    receiver_account.nonce += 1;
    receiver_account.free = receiver_account.free.saturating_add(amount);
    receiver_account.write_to_storage(asset_id_bytes, receiver_account_bytes);
}

fn extract_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().unwrap())
}

fn extract_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().unwrap())
}
