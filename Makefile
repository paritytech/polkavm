SERVICES = bootstrap tribonacci megatron transfer balances algo null_authorizer auth_copy blake2b fib corevm corevm_child game_of_life game_of_life_child game_of_life_parent_only revm_test compress game_of_life_manifest doom hello_world mnist

TARGET_DIR = riscv64emac-unknown-none-polkavm/release

.PHONY: all $(SERVICES) clean

all: $(SERVICES)

$(SERVICES):
	clear
	@echo "Building $@ service..."
	cd ./services/$@ && cargo build --release --verbose --target-dir ./target
	cd ../../
	@if [ "$@" = "null_authorizer" ]; then \
		cargo run -p polkatool jam-service services/$@/target/$(TARGET_DIR)/$@ -o services/$@/$@.pvm -d services/$@/$@_blob.pvm -i; \
	elif echo "$@" | grep -q '_child$$'; then \
		cargo run -p polkatool jam-service services/$@/target/$(TARGET_DIR)/$@ -o services/$@/$@.pvm -d services/$@/$@_blob.pvm -m; \
	else \
		cargo run -p polkatool jam-service services/$@/target/$(TARGET_DIR)/$@ -o services/$@/$@.pvm -d services/$@/$@_blob.pvm; \
	fi
	cargo run -p polkatool disassemble services/$@/$@_blob.pvm --show-raw-bytes > ./services/$@/$@.txt
	@echo "Service $@ built successfully!"

clean:
	@echo "Cleaning up all build artifacts..."
	rm -rf services/*/target
	rm -rf services/*/*.pvm
	rm -rf services/*/*.txt
	@echo "Cleanup complete!"

setup_toolchain:
	clear
	@if [ -d ~/.rustup/toolchains/rve-nightly ]; then \
		echo "Toolchain ~/.rustup/toolchains/rve-nightly already exists, skipping download and installation."; \
	else \
		echo "Downloading and installing paritytech rve-nightly..."; \
		wget https://github.com/paritytech/rustc-rv32e-toolchain/releases/download/v1.1.0/rust-rve-nightly-2024-01-05-x86_64-unknown-linux-gnu.tar.zst; \
		tar --zstd -xf rust-rve-nightly-2024-01-05-x86_64-unknown-linux-gnu.tar.zst; \
		mv rve-nightly ~/.rustup/toolchains/; \
		rm rust-rve-nightly-2024-01-05-x86_64-unknown-linux-gnu.tar.zst; \
	fi

	@if [ -d ~/.rustup/toolchains/rve-nightly-186 ]; then \
		echo "Toolchain ~/.rustup/toolchains/rve-nightly-186 already exists, skipping download and installation."; \
	else \
		clear; \
		echo "Downloading and installing clw8998 rve-nightly-186..."; \
		wget https://github.com/clw8998/rustc-rv32e-toolchain/releases/download/v1.2.0/rust-rve-1.86.0-dev-2025-03-31-x86_64-unknown-linux-gnu.tar.zst; \
		tar --zstd -xf rust-rve-1.86.0-dev-2025-03-31-x86_64-unknown-linux-gnu.tar.zst; \
		mv rve-nightly rve-nightly-186; \
		mv rve-nightly-186 ~/.rustup/toolchains/; \
		rm rust-rve-1.86.0-dev-2025-03-31-x86_64-unknown-linux-gnu.tar.zst; \
	fi

	@echo "Toolchain setup complete. You may need to run 'rustup show' to verify the installed toolchains ('rve-nightly' and 'rve-nightly-186')."
	rustup show
