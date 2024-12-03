# JAM Service Setup Guide (FIB)

## Step 1: Download and Set Up the Toolchain

1. **Download the Toolchain File**  
   Visit [Parity Tech Rustc RV32E Toolchain Releases](https://github.com/paritytech/rustc-rv32e-toolchain/releases) and download the appropriate toolchain for your system.

2. **Extract and Move the Toolchain**  
   Extract the toolchain file with `zstd` compression and move it to Rustup's toolchain directory:
   ```bash
   tar --zstd -xf rust-rve-nightly-2024-01-05-x86_64-unknown-linux-gnu.tar.zst
   mv rve-nightly ~/.rustup/toolchains/
   ```
   If you encounter errors with the above commands, you might need to install `zstd`:
   ```bash
   sudo apt update
   sudo apt install zstd
   ```

3. **Verify Installation**  
   Check if the toolchain is correctly installed:
   ```bash
   ls ~/.rustup/toolchains/
   ```
   Ensure `rve-nightly` is listed among the installed toolchains.

4. **Set Default Toolchain**  
   Set `rve-nightly` as the default toolchain:
   ```bash
   echo "export RUSTUP_TOOLCHAIN=rve-nightly" >> ~/.bashrc
   source ~/.bashrc
   ```
---

## Step 2: Build the JAM FIB Service

1. **Navigate to the Service Directory**  
   ```bash
   cd services/fib
   ```

2. **Build the Service in Release Mode**  
   ```bash
   cargo build --release --target-dir ./target
   ```
   > **Note:** If you encounter an error related to OpenSSL (e.g., `version OPENSSL_1_1_1' not found`), please refer to [Optional: Fix OpenSSL Version Error](#Optional-Fix-OpenSSL-Version-Error-version-OPENSSL_1_1_1-not-found) at the end of this guide.

---

## Step 3: Build the Polkatool

1. **Move to the Polkatool Directory**  
   ```bash
   cd ../../tools/polkatool
   ```

2. **Build Polkatool in Release Mode**  
   ```bash
   cargo build --release --target-dir ./target
   ```
   > **Note:** If you encounter an error related to OpenSSL (e.g., `version OPENSSL_1_1_1' not found`), please refer to [Optional: Fix OpenSSL Version Error](#Optional-Fix-OpenSSL-Version-Error-version-OPENSSL_1_1_1-not-found) at the end of this guide.

---

## Step 4: Generate the FIB Service and Blob

1. **Return to the `polkavm` Root Directory**  
   Before generating the FIB service and blob file, return to the `polkavm` root directory:
   ```bash
   cd ../../
   ```

2. **Generate the JAM FIB Service and Blob**  
   Use `polkatool` to generate the JAM FIB service and blob file:

   To generate `32-bit` version blob, set: `riscv32emac-unknown-none-polkavm/release/fib` and `-b 32` :
   ```bash
    cargo run -p polkatool jam-service services/fib/target/riscv32emac-unknown-none-polkavm/release/fib -o services/fib/fib.pvm -d services/fib/blob.pvm -b 32
   ```

   To generate `64-bit` version blob, set: `riscv64emac-unknown-none-polkavm/release/fib` and `-b 64` :
   ```bash
    cargo run -p polkatool jam-service services/fib/target/riscv64emac-unknown-none-polkavm/release/fib -o services/fib/fib.pvm -d services/fib/blob.pvm -b 64
   ```

3. **Generated Output Files**  
   After running the above command, two files will be created:
   - `fib_32.pvm`: JAM-ready top-level service blob. (`32-bit` ver)
   - `blob_32.pvm`: This file can be disassembled with `polkatool`. (`32-bit` ver)

4. **Disassemble the Code**  
   To compile and disassemble the code, use:
   ```bash
   cargo run -p polkatool disassemble services/fib/blob_32.pvm --show-raw-bytes
   ```

---

## Step 5: Create a New Service Similar to FIB

1. **Navigate to the Services Directory**  
   ```bash
   cd services
   ```

2. **Add New Service to Root `Cargo.toml`**  
   Modify the `Cargo.toml` file in the `services` directory to add your new service under `[workspace]` members:

   ```toml
   [profile.release]
   lto = "fat"
   panic = "abort"
   opt-level = 3
   codegen-units = 1
   debug = true

   [profile.no-lto]
   inherits = "release"
   lto = false

   [workspace]
   resolver = "2"
   members = [
      "fib",
      "admin",
      "staking",
      "accounts", 
      "your_new_service",
   ]
   ```

3. **Create the New Service**  
   Replace `your_new_service` with your desired service name:
   ```bash
   cargo new your_new_service --bin --vcs none
   ```

4. **Navigate to Your New Service Directory**  
   ```bash
   cd your_new_service
   ```

5. **Modify `main.rs`**  
   Edit `./src/main.rs` to define your new service.

6. **Adjust `Cargo.toml` in the New Service Directory**  
   Update `./Cargo.toml` in the new service directory to ensure the package name and library name reflect `your_new_service`:

   ```toml
   [package]
   name = "your_new_service"
   version = "0.1.0"
   edition = "2024"
   publish = false

   [lib]
   name = "your_new_service"
   path = "src/main.rs"
   crate-type = ["staticlib", "rlib"]

   [dependencies]
   polkavm-derive = { path = "../../crates/polkavm-derive" }
   simplealloc = { path = "../../crates/simplealloc" }
   ```

7. **Build the New Service in Release Mode**  
   ```bash
   cargo build --release --target-dir ./target
   ```

8. **Return to `polkavm` Root Directory**  
   Before generating the new service and blob file, return to the `polkavm` root directory:
   ```bash
   cd ../../
   ```

9. **Generate the Service and Blob Files**  
   Use `polkatool` to generate the new service and blob file:
   ```bash
   cargo run -p polkatool jam-service services/your_new_service/target/riscv32ema-unknown-none-elf/release/your_new_service -o services/your_new_service/your_new_service.pvm -d services/your_new_service/blob.pvm
   ```

10. **Disassemble the Service Code**  
    Disassemble the newly created service:
    ```bash
    cargo run -p polkatool disassemble services/your_new_service/blob.pvm --show-raw-bytes
    ```

---

## Optional: Download Generated `.pvm` Files to Local

1. **Locate the `.pvm` File Path**  
   For example, if the `.pvm` file is located in `~/polkavm/services/your_new_service/`, the path would be:
   ```bash
   ~/polkavm/services/your_new_service/your_new_service.pvm
   ```

2. **Use SCP to Download the File**  
   In your local terminal, use `scp` to download the file. Replace `username` and `remote_ip` with your server’s credentials:
   ```bash
   scp username@remote_ip:~/polkavm/services/your_new_service/your_new_service.pvm .
   ```

3. **Verify Downloaded File**  
   Check the local directory to ensure the `.pvm` file is present:
   ```bash
   ls your_new_service.pvm
   ```

---

## Optional: Fix OpenSSL Version Error (`version OPENSSL_1_1_1' not found`)

If you encounter the `version OPENSSL_1_1_1' not found` error, follow these steps:

### Installation Steps

1. **Download the `libssl1.1` package**  
   ```bash
   wget http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb
   ```

2. **Install the downloaded package**  
   ```bash
   sudo dpkg -i libssl1.1_1.1.1f-1ubuntu2_amd64.deb
   ```

3. **Fix dependencies (if any errors occur)**  
   ```bash
   sudo apt-get install -f
   ```
--- 

## FIB Code Explanation

### Initialize a Buffer

The program starts by initializing a buffer to hold the Fibonacci calculation data. It is 12 bytes long, representing the three numbers required by the FIB array (each stored in 4 bytes):

```rust=
let mut buffer = [0u8; 12];
```

### Import Data into the Buffer

The `import` (host import) function is used to populate the buffer. Here:
- `0` represents the first segment.
- `buffer.as_mut_ptr()` points to the start of the buffer in RAM.
- `buffer.len()` specifies the data length.

```rust=
let result = unsafe { import(0, buffer.as_mut_ptr(), buffer.len() as u32) };
```

### Parse and Calculate the Fibonacci Term

If the data import is successful (i.e., `result` is `0`), the program:
1. Extracts little-endian data from the buffer and converts it to `u32`.
2. Calculates the next term in the Fibonacci sequence by adding the previous two terms.

```rust=
let n = u32::from_le_bytes(buffer[0..4].try_into().unwrap());
let fib_n = u32::from_le_bytes(buffer[4..8].try_into().unwrap());
let fib_n_minus_1 = u32::from_le_bytes(buffer[8..12].try_into().unwrap());

let new_fib_n = fib_n + fib_n_minus_1;
```

### Store Results Back in the Buffer

The calculated result is converted back into little-endian bytes and stored in the buffer.

```rust=
buffer[0..4].copy_from_slice(&(n + 1).to_le_bytes());
buffer[4..8].copy_from_slice(&new_fib_n.to_le_bytes());
buffer[8..12].copy_from_slice(&fib_n.to_le_bytes());
```

### Export the Buffer Data

Finally, the buffer is exported with the `export` (host export) function, using the buffer’s pointer and length.

```rust=
unsafe {
    export(buffer.as_mut_ptr(), buffer.len() as u32);
}
```

### Store buffer's info

Store the starting position of the buffer in `a3` and the buffer length in `a4` to ensure the PVM output argument can be retrieved.

```rust=
let buffer_addr = buffer.as_ptr() as u32;
let buffer_len = buffer.len() as u32;
unsafe {
    core::arch::asm!(
        "mv a3, {0}",
        "mv a4, {1}",
        in(reg) buffer_addr,
        in(reg) buffer_len,
    );
}
```



## Complete Code

Below is the full Rust code for the Fibonacci sequence calculation:

```rust=
#[polkavm_derive::polkavm_export]
extern "C" fn is_authorized() -> u32 {


    0
}

#[polkavm_derive::polkavm_export]
extern "C" fn refine() -> u32 {
    let mut buffer = [0u8; 12];
    let result = unsafe { import(0, buffer.as_mut_ptr(), buffer.len() as u32) };

    if result == 0 {
        let n = u32::from_le_bytes(buffer[0..4].try_into().unwrap());
        let fib_n = u32::from_le_bytes(buffer[4..8].try_into().unwrap());
        let fib_n_minus_1 = u32::from_le_bytes(buffer[8..12].try_into().unwrap());
    
        let new_fib_n = fib_n + fib_n_minus_1;
    
        buffer[0..4].copy_from_slice(&(n + 1).to_le_bytes());
        buffer[4..8].copy_from_slice(&new_fib_n.to_le_bytes());
        buffer[8..12].copy_from_slice(&fib_n.to_le_bytes());
    } else {
        buffer.copy_from_slice(&[1u8, 0u8, 0u8, 0u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8]);
    }

    unsafe {
        export(buffer.as_mut_ptr(), buffer.len() as u32);
    }
    let buffer_addr = buffer.as_ptr() as u32;
    let buffer_len = buffer.len() as u32;
    unsafe {
        core::arch::asm!(
            "mv a3, {0}",
            "mv a4, {1}",
            in(reg) buffer_addr,
            in(reg) buffer_len,
        );
    }
    0
}

#[polkavm_derive::polkavm_export]
extern "C" fn accumulate() -> u32 {
    let buffer = [0u8; 12];
    let key = [0u8; 1];

    unsafe {
        write(key.as_ptr(), 1, buffer.as_ptr(), buffer.len() as u32);
    }

    0
}

#[polkavm_derive::polkavm_export]
extern "C" fn on_transfer() -> u32 {
    0
}
```