default?=bootstrap

setup_service:
	cd $(default) && \
	cargo build --release --target-dir ./target && \
	cd ../../ && \
	cargo run -p polkatool jam-service services/$(default)/target/riscv64emac-unknown-none-polkavm/release/$(default) -o services/$(default)/$(default).pvm -d services/$(default)/$(default)_blob.pvm && \
	cargo run -p polkatool disassemble services/$(default)/$(default)_blob.pvm --show-raw-bytes > ./services/$(default)/$(default).txt
